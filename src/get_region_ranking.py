#import dependencies

import time
import pandas as pd
from pygbif import occurrences
import pycountry
from tqdm.auto import tqdm

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
    Ranks potential IAS threats locally first, then validates only the highest-risk 
    species against GBIF using a tqdm progress bar, respecting API rate limits.
    """
    # 1. Clean Data
    df['location_clean'] = df['location'].dropna().astype(str).str.strip().str.title()
    df['taxon_clean'] = df['taxon'].dropna().astype(str).str.strip().str.lower()
    target_region = target_region.strip().title()
    
    # Try to resolve country code
    gbif_country_code = None
    try:
        gbif_country_code = pycountry.countries.lookup(target_region).alpha_2
    except LookupError:
        pass

    # 2. Local exclusions (Phase 1)
    target_species_clean = set(df[df['location_clean'] == target_region]['taxon_clean'])
    is_introduced = df['establishmentMeans'].astype(str).str.strip().str.lower() == 'introduced'
    ias_df = df[is_introduced].dropna(subset=['taxon_clean', 'location_clean'])
    
    new_ias_df = ias_df[~ias_df['taxon_clean'].isin(target_species_clean)]
    
    if new_ias_df.empty:
        return "No new invasive species found."

    taxon_name_map = new_ias_df.groupby('taxon_clean')['taxon'].first().to_dict()
    sim_dict = dict(zip(similarity_df['Region'], similarity_df['Similarity score']))
    grouped_ias = new_ias_df.groupby('taxon_clean')['location_clean'].apply(set)
    
    # 3. RANK FIRST (Calculate scores locally)
    print("Ranking species locally first...")
    risk_records = []
    for clean_species, regions in grouped_ias.items():
        scores = [sim_dict.get(region, 0.0) for region in regions]
        if scores:
            risk_records.append({
                'clean_taxon': clean_species,
                'Species': taxon_name_map[clean_species],
                'Cumulative risk score': sum(scores),
                'Max single-region similarity': max(scores),
                'Found in regions': ", ".join(sorted(regions)),
                'Region count': len(regions)
            })
            
    risk_df = pd.DataFrame(risk_records)
    risk_df = risk_df.sort_values(
        by=['Cumulative risk score', 'Max single-region similarity'], 
        ascending=[False, False]
    ).reset_index(drop=True)

    # 4. VALIDATE LATER (Check only the top K against GBIF)
    if not gbif_country_code:
        print("No country code resolved. Returning purely local rankings.")
        return risk_df.drop(columns=['clean_taxon'])

    print(f"Validating top threats against GBIF for '{gbif_country_code}'...")
    
    validated_records = []
    api_calls_made = 0
    errors_hit = 0
    
    # Initialize tqdm progress bar
    with tqdm(total=species_to_validate, desc="True threats found", unit="species") as pbar:
        for _, row in risk_df.iterrows():
            # Stop once we've found our target amount of validated threats
            if len(validated_records) >= species_to_validate:
                break
                
            original_sp = row['Species']
            
            # THROTTLE: Respect rate limits before making the call
            if api_calls_made > 0:
                time.sleep(sleep_time)
                
            api_calls_made += 1
            
            try:
                res = occurrences.search(scientificName=original_sp, country=gbif_country_code, limit=1)
                
                if res['count'] == 0:
                    # Genuinely missing from GBIF! Add it and update the progress bar.
                    validated_records.append(row)
                    pbar.update(1)
                
                # Update the side-counter
                pbar.set_postfix({'API Calls': api_calls_made, 'Errors': errors_hit})
                        
            except Exception as e:
                errors_hit += 1
                # Back off dynamically if we hit an error (e.g., HTTP 429 Too Many Requests)
                time.sleep(sleep_time * 5) 
                validated_records.append(row) # Keeping it to be safe
                pbar.update(1)
                pbar.set_postfix({'API Calls': api_calls_made, 'Errors': errors_hit})

    print(f"Done! Evaluated {api_calls_made} species via GBIF to isolate {len(validated_records)} validated threats.")
    
    final_df = pd.DataFrame(validated_records)
    if not final_df.empty:
        final_df = final_df.drop(columns=['clean_taxon']).reset_index(drop=True)
        
    return final_df