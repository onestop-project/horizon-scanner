#import dependencies

import time
import pandas as pd
from pygbif import occurrences
import pycountry
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm.auto import tqdm
from geopy.distance import geodesic
from pygbif import species as gbif_species
import wbgapi as wb
import country_converter as coco
import requests
import logging

#get country centroids

def get_region_centroids(regions_list):
    """
    Fetches (latitude, longitude) centroids for a list of unique regions.
    Uses rate limiting to respect OpenStreetMap API guidelines.
    """
    geolocator = Nominatim(user_agent="ias_spatial_assessor")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
    
    centroids = {}
    print(f"Fetching geographic centroids for {len(regions_list)} unique regions...")
    
    for region in tqdm(regions_list, desc="Geocoding Regions"):
        try:
            location = geocode(region)
            if location:
                centroids[region] = (location.latitude, location.longitude)
            else:
                centroids[region] = None
        except Exception:
            centroids[region] = None 
            
    return centroids

#get country similarity rankings for a target region

def rank_regions_by_similarity(df, target_region, compare_only_introduced=False):
    """
    Ranks all regions in the dataframe based on species similarity to a target region.
    
    Parameters:
    - df: The biodiversity dataframe.
    - target_region: The name of the location to compare others against (e.g., 'Aegean').
    - compare_only_introduced: If True, only compares species where establishmentMeans == 'introduced'.
    """
    
    # Create a working copy to avoid altering your original dataframe
    working_df = df.copy()
    
    # Clean the location names
    working_df['location_clean'] = working_df['location'].dropna().astype(str).str.strip().str.title()
    target_region = target_region.strip().title()
    
    if target_region not in working_df['location_clean'].values:
        raise ValueError(f"'{target_region}' was not found in the location column.")
        
    # Filter for introduced species if requested
    if compare_only_introduced:
        working_df = working_df[working_df['establishmentMeans'].str.lower() == 'introduced']
        
    # Drop rows missing either taxon or location
    clean_data = working_df.dropna(subset=['taxon', 'location_clean'])
    
    # Group by region and create a set of species (taxons) for each
    region_species = clean_data.groupby('location_clean')['taxon'].apply(set).to_dict()
    
    # Retrieve the target region's species list
    # (Use .get() with an empty set just in case the target region has no species after filtering)
    target_species = region_species.get(target_region, set())
    
    if not target_species:
        return f"No species found for '{target_region}' with the current filters."
    
    results = []
    for region, species_set in region_species.items():
        if region == target_region:
            continue
            
        intersection = len(target_species.intersection(species_set))
        union = len(target_species.union(species_set))
        
        similarity_score = (intersection / union) if union > 0 else 0.0
        
        results.append({
            'Region': region,
            'Shared species': intersection,
            'Total unique species in both': union,
            'Target region total': len(target_species),
            'Comparison region total': len(species_set),
            'Similarity score': similarity_score
        })
        
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        return "No other regions available for comparison."
        
    ranked_results = results_df.sort_values(by='Similarity score', ascending=False).reset_index(drop=True)
    return ranked_results

# Generate probability risk based on which IAS come from similar countries but have not yet invaded BE

def predict_ias_risk(df, similarity_df, target_region, species_to_validate, sleep_time=0.5):
    """
    Ranks potential IAS threats using a Cumulative Hybrid Score.
    Validates presence via GBIF and includes the Backbone matched name for manual verification.
    """
    # 1. Setup & Cleaning
    df['location_clean'] = df['location'].dropna().astype(str).str.strip().str.title()
    df['taxon_clean'] = df['taxon'].dropna().astype(str).str.strip().str.lower()
    target_region = target_region.strip().title()
    
    gbif_country_code = None
    try:
        gbif_country_code = pycountry.countries.lookup(target_region).alpha_2
    except LookupError:
        pass

    # 2. Local Exclusions (Phase 1)
    target_species_clean = set(df[df['location_clean'] == target_region]['taxon_clean'])
    is_introduced = df['establishmentMeans'].astype(str).str.strip().str.lower() == 'introduced'
    ias_df = df[is_introduced].dropna(subset=['taxon_clean', 'location_clean'])
    new_ias_df = ias_df[~ias_df['taxon_clean'].isin(target_species_clean)]
    
    if new_ias_df.empty:
        return "No new invasive species found."

    taxon_name_map = new_ias_df.groupby('taxon_clean')['taxon'].first().to_dict()
    
    # Identify similarity column (case-insensitive)
    sim_col = [c for c in similarity_df.columns if 'imilarity' in c]
    sim_dict = dict(zip(similarity_df['Region'], similarity_df[sim_col]))
    
    grouped_ias = new_ias_df.groupby('taxon_clean')['location_clean'].apply(set)
    extracted_centroids = df.dropna(subset=['location_clean', 'Centroid']).set_index('location_clean')['Centroid'].to_dict()
    target_coords = extracted_centroids.get(target_region)

    # 3. Hybrid Ranking (Phase 2)
    print("Ranking species locally...")
    risk_records = []
    for clean_species, regions in grouped_ias.items():
        hybrid_cumulative_score = 0.0
        max_sim = 0.0
        for region in regions:
            sim = sim_dict.get(region, 0.0)
            region_coords = extracted_centroids.get(region)
            if sim > max_sim: max_sim = sim
            
            if target_coords and region_coords:
                dist_km = geodesic(target_coords, region_coords).kilometers
                # 1000km half-life decay
                weight = 1.0 / (1.0 + (dist_km / 1000.0))
            else:
                weight = 0.05
            hybrid_cumulative_score += (sim * weight)
            
        risk_records.append({
            'clean_taxon': clean_species,
            'Species': taxon_name_map[clean_species],
            'Hybrid Risk Score': hybrid_cumulative_score, 
            'Max Single-Region Similarity': max_sim,
            'Found In Regions': ", ".join(sorted(regions)),
            'Region Count': len(regions)
        })
            
    risk_df = pd.DataFrame(risk_records).sort_values(
        by=['Hybrid Risk Score', 'Max Single-Region Similarity'], 
        ascending=[False, False]
    ).reset_index(drop=True)

    # 4. GBIF Validation (Phase 3)
    if not gbif_country_code:
        return risk_df.drop(columns=['clean_taxon'])

    print(f"Validating top threats against GBIF for '{gbif_country_code}'...")
    validated_records = []
    api_calls_made = 0
    errors_hit = 0
    
    with tqdm(total=species_to_validate, desc="True threats found", unit="species") as pbar:
        for _, row in risk_df.iterrows():
            if len(validated_records) >= species_to_validate:
                break
                
            if api_calls_made > 0:
                time.sleep(sleep_time)
            
            api_calls_made += 1
            current_sp = row['Species']
            
            try:
                # Get Backbone Match for the column
                # species is the module imported from pygbif
                match = match = gbif_species.name_backbone(name=current_sp)
                gbif_name = match.get('scientificName', 'No match')
                match_type = match.get('matchType', 'NONE')
                
                # Check for occurrences in target country
                res = occurrences.search(scientificName=current_sp, country=gbif_country_code, limit=0)
                
                if res['count'] == 0:
                    # Species not found in target country: Add to horizon scan list
                    new_row = row.to_dict()
                    new_row['GBIF Matched Name'] = f"{gbif_name} [{match_type}]"
                    validated_records.append(new_row)
                    pbar.update(1)
                
                pbar.set_postfix({'API Calls': api_calls_made, 'Errors': errors_hit})
                        
            except Exception as e:
                errors_hit += 1
                time.sleep(sleep_time * 5)
                # On error, we keep it as a potential threat but flag the name
                err_row = row.to_dict()
                err_row['GBIF Matched Name'] = "FETCH ERROR"
                validated_records.append(err_row)
                pbar.update(1)

    final_df = pd.DataFrame(validated_records)
    if not final_df.empty:
        final_df = final_df.drop(columns=['clean_taxon'], errors='ignore').reset_index(drop=True)
        
    return final_df

def get_valid_countries(df, loc_col='location'):
    """Converts unique locations to ISO3 with a progress bar."""
    cc = coco.CountryConverter()
    unique_locs = df[loc_col].unique()
    
    iso_map = {}
    for loc in tqdm(unique_locs, desc="Step 1/4: Validating Locations"):
        # We do this one by one to keep the tqdm bar moving
        iso = cc.convert(names=loc, to='ISO3', not_found=None)
        iso_map[loc] = iso
    
    df['iso3'] = df[loc_col].map(iso_map)
    # Filter out 'not found' or None
    valid_df = df[(df['iso3'] != 'not found') & (df['iso3'].notna())].copy()
    return valid_df

def fetch_wdi_data(iso_list):
    """Fetches World Bank indicators with robust column naming."""
    indicators = {'NY.TRD.TICR.ZS.GD': 'trade_pct_gdp', 'EN.POP.DNST': 'pop_density'}
    all_results = []
    
    chunk_size = 10
    chunks = [iso_list[i:i + chunk_size] for i in range(0, len(iso_list), chunk_size)]
    
    for chunk in tqdm(chunks, desc="Step 2/4: Fetching Economic Data"):
        try:
            # Note: mrv=1 returns a DataFrame with ISO codes as the index
            data = wb.data.DataFrame(list(indicators.keys()), chunk, mrv=1)
            if not data.empty:
                all_results.append(data)
            time.sleep(1.2)
        except Exception as e:
            print(f"Error in WDI chunk {chunk}: {e}")
            
    if not all_results:
        return pd.DataFrame(columns=['economy', 'trade_pct_gdp', 'pop_density'])
        
    # Concatenate and force the index to be named 'economy'
    final_wdi = pd.concat(all_results)
    final_wdi = final_wdi.rename(columns=indicators)
    
    # Use 'names' parameter to ensure the column is called 'economy'
    final_wdi = final_wdi.reset_index(names='economy') 
    
    return final_wdi

def fetch_cckp_climate(iso, var='tas'):
    """Internal helper for CCKP."""
    url = f"https://climateknowledgeportal.worldbank.org/api/data/get-aggregated-data/cru-x0.5/{var}/climatology/annual/{iso}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['data']['value'] if data else None
    except: return None

def enrich_climate_and_gbif(df):
    """Fetches Climate and GBIF data with a combined progress bar."""
    unique_isos = df['iso3'].unique().tolist()
    temp_cache, precip_cache, ias_cache = {}, {}, {}

    for iso in tqdm(unique_isos, desc="Step 3/4: Fetching Climate & GBIF"):
        # CCKP Climate (Temp & Precip)
        temp_cache[iso] = fetch_cckp_climate(iso, 'tas')
        time.sleep(0.4) # Small sleep between different API endpoints
        precip_cache[iso] = fetch_cckp_climate(iso, 'pr')
        
        # GBIF Invasion Counts
        try:
            time.sleep(0.5)
            ias_cache[iso] = gbif_species.name_count(is_invasive=True, country=iso[:2])
        except:
            ias_cache[iso] = 0
            
    df['annual_mean_temp'] = df['iso3'].map(temp_cache)
    df['annual_precip'] = df['iso3'].map(precip_cache)
    df['invasion_debt_count'] = df['iso3'].map(ias_cache)
    return df

def full_enrichment_pipeline(df, worldclim_config=None):
    """The master function calling everything with bars."""
    # 1. Locations
    df = get_valid_countries(df)
    
    # 2. World Bank WDI
    unique_isos = df['iso3'].unique().tolist()
    wdi_df = fetch_wdi_data(unique_isos)
    df = df.merge(wdi_df, left_on='iso3', right_on='economy', how='left')
    
    # 3. Climate & GBIF
    df = enrich_climate_and_gbif(df)