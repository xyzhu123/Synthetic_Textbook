import os
import json
import csv
import argparse
import string
from tqdm import tqdm
from typing import List, Dict, Any, Optional


def get_client(provider: str, api_key: str):
    if provider == 'openai':
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    elif provider == 'together':
        from together import Together
        return Together(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_message(role: str, content: str) -> Dict[str, Any]:
    return {
        "role": role,
        "content": [{"type": "text", "text": content}]
    }


def get_completion(client, 
                   messages: List[Dict], 
                   model_name: str, 
                   max_tokens: int, 
                   temperature: float, 
                   provider: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        **({"response_format": {"type": "text"}} if provider == 'openai' else {})
    )
    return response.choices[0].message.content


def generate_subfields(client, 
                       model_name: str, 
                       provider: str, 
                       keyword: str, 
                       num_subfields: int,
                       temperature: float, 
                       max_tokens: int) -> List[str]:
    print(f"\n{'='*60}")
    print(f"STAGE 1: Generating {num_subfields} subfields for '{keyword}'")
    print(f"{'='*60}\n")
    
    sys_p = f"Assume you are an expert and seasoned scholar with 20+ years of academic experience in {keyword}."
    user_p = f'Give me the names of {num_subfields} subfields of {keyword}, separate the names with ",". Directly output the names without any other text.'
    
    sys_msg = create_message('system', sys_p)
    usr_msg = create_message('user', user_p)
    messages = [sys_msg, usr_msg]
    
    response = get_completion(client, messages, model_name, max_tokens, temperature, provider)
    
    subfields = response.split(",")
    subfields = [x.strip().translate(str.maketrans('', '', string.punctuation)) for x in subfields]
    
    print(f"Generated subfields:")
    for i, subfield in enumerate(subfields, 1):
        print(f"  {i}. {subfield}")
    
    return subfields


def generate_bullet_points(client, 
                           model_name: str, 
                           provider: str,
                           keyword: str, 
                           subfields: List[str],
                           audiences: List[str], 
                           data_path: str,
                           max_bullets: int, 
                           temperature: float,
                           max_tokens: int) -> str:
    print(f"\n{'='*60}")
    print(f"STAGE 2: Generating bullet points")
    print(f"{'='*60}\n")
    
    normalized_model = model_name.split('/')[-1]
    
    sys_p_template = "Assume you are an expert and seasoned scholar with 20+ years of academic experience in {field}."
    user_bullet_template = (
        "You are writing an engaging, accessible, and age-appropriate textbook on {subfield} "
        "within {field} for {audience}. Craft a comprehensive set of bullet points that covers "
        "essential scientific concepts and practical applications. The bullet points should be "
        "clear, non-hierarchical, and self-contained, aligning with {audience}s' knowledge level "
        "and ensuring accessibility and relevance. Start each bullet point with '>>'. "
        "Don't include any '\\n' in the bullet points. At most 20 bullet points in total, "
        "don't include any other text."
    )
    
    os.makedirs(data_path, exist_ok=True)
    all_bullet_points = []
    total_combinations = len(subfields) * len(audiences)
    
    with tqdm(total=total_combinations, desc="Generating bullet points") as pbar:
        for subfield in subfields:
            for audience in audiences:
                sys_p = sys_p_template.format(field=keyword)
                user_bullet_p = user_bullet_template.format(
                    field=keyword, 
                    subfield=subfield, 
                    audience=audience
                )
                
                sys_msg = create_message('system', sys_p)
                usr_msg = create_message('user', user_bullet_p)
                messages = [sys_msg, usr_msg]
                
                response = get_completion(client, messages, model_name, max_tokens, temperature, provider)
                
                bullet_points = response.split('>>')
                bullet_points = [point.replace('\n', '').strip() for point in bullet_points]
                bullet_points = [point for point in bullet_points if len(point)]
                bullet_points = bullet_points[:max_bullets]
                
                entry = {
                    'field': keyword,
                    'subfield': subfield,
                    'audience': audience,
                    'bullet_points': bullet_points
                }
                all_bullet_points.append(entry)
                pbar.update(1)
    
    output_file = f"{data_path}/{keyword}_{normalized_model}_bps.json"
    with open(output_file, 'w') as json_file:
        json.dump(all_bullet_points, json_file, indent=4)
    
    print(f"\nGenerated bullet points for {len(all_bullet_points)} subfield-audience combinations")
    print(f"Saved to {output_file}")
    
    return output_file


def generate_chapters(client, 
                      model_name: str, 
                      provider: str,
                      keyword: str, 
                      data_path: str, 
                      num_chapters_per_bp: int,
                      temperature: float, 
                      max_tokens: int) -> str:
    print(f"\n{'='*60}")
    print(f"STAGE 3: Generating chapters from bullet points")
    print(f"{'='*60}\n")
    
    normalized_model = model_name.split('/')[-1]
    
    sys_p_template = "Assume you are an expert and seasoned scholar with 20+ years of academic experience in {field}."
    user_chapter_template = (
        "You are writing an engaging, accessible, and age-appropriate textbook on {subfield} "
        "within {field} for {audience}. Please write a comprehensive chapter based on the "
        "bullet point \"{bulletp}\". \nThe chapter should be: \n"
        "a. Self-contained, covering the topic comprehensively with long paragraphs to present everything you know. \n"
        "b. On topic with respect to the bullet point you are currently writing on. \n"
        "c. Define key terms clearly using words, symbolic axioms, or both. \n"
        "d. Adjust the writing to match the audience's knowledge level and understanding."
    )
    
    bps_file = f'{data_path}/{keyword}_{normalized_model}_bps.json'
    
    if not os.path.exists(bps_file):
        print(f"Error: Bullet points file not found: {bps_file}")
        print(f"Please run with --stages bulletpoints first.")
        return None
    
    with open(bps_file, 'r') as f:
        all_bps_data = json.load(f)
    
    total = sum(len(entry['bullet_points']) for entry in all_bps_data)
    
    print(f"Loaded {len(all_bps_data)} subfield-audience combinations")
    print(f"Total bullet points: {total}")
    print(f"Generating {num_chapters_per_bp} chapter(s) per bullet point")
    
    generated = []
    with tqdm(total=total * num_chapters_per_bp, desc="Generating chapters") as pbar:
        for entry in all_bps_data:
            field = entry['field']
            subfield = entry['subfield']
            audience = entry['audience']
            bulletps = entry['bullet_points']
            
            sys_p = sys_p_template.format(field=field)
            
            for bulletp in bulletps:
                user_chapter_p = user_chapter_template.format(
                    field=field,
                    subfield=subfield,
                    audience=audience,
                    bulletp=bulletp
                )
                
                for i in range(num_chapters_per_bp):
                    sys_msg = create_message('system', sys_p)
                    usr_msg = create_message('user', user_chapter_p)
                    messages = [sys_msg, usr_msg]
                    
                    model_p = get_completion(
                        client,
                        messages,
                        model_name,
                        max_tokens,
                        temperature,
                        provider
                    )
                    
                    now_generated = {
                        'field': field,
                        'subfield': subfield,
                        'audience': audience,
                        'bulletp': bulletp,
                        'response': model_p
                    }
                    generated.append(now_generated)
                    pbar.update(1)
    
    output_file = f'{data_path}/{keyword}_{normalized_model}_textbook_generated.json'
    with open(output_file, 'w') as f:
        json.dump(generated, f, indent=4)
    
    print(f'\nGenerated {len(generated)} total chapters')
    print(f'Saved to {output_file}')
    
    return output_file


def postprocess_chapters(input_file: str, 
                         output_file: str, 
                         threshold: int, 
                         show_stats: bool = False) -> str:
    print(f"\n{'='*60}")
    print(f"STAGE 4: Post-processing chapters")
    print(f"{'='*60}\n")
    
    def clean_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        r = row['response']
        r = r.replace('*', '')
        r = r.replace('#', '')
        r = r.split('\n')
        r = [line.strip() for line in r]
        r = [' '.join(line.split()) for line in r]
        r = [line for line in r if len(line) > 0]
        
        new_rows = []
        metas = {k: v for k, v in row.items() if k != 'response'}
        for line in r:
            now_row = metas.copy()
            now_row['response'] = line
            new_rows.append(now_row)
        return new_rows
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} chapters")
    
    print("Cleaning and splitting into paragraphs...")
    cleaned = [
        flat_row
        for row in raw
        for flat_row in clean_row(row)
    ]
    print(f"Total paragraphs after splitting: {len(cleaned)}")
    
    if show_stats:
        freq = {}
        for row in cleaned:
            word_count = len(row['response'].split(' '))
            freq[word_count] = freq.get(word_count, 0) + 1
        print(f"Word count range: {min(freq.keys())} - {max(freq.keys())}")
    
    print(f"Filtering paragraphs with word count >= {threshold}...")
    filtered = [
        row for row in cleaned
        if len(row['response'].split(' ')) >= threshold
    ]
    print(f"Paragraphs after filtering: {len(filtered)}")
    
    print(f"Saving to {output_file}...")
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text'])
        for row in filtered:
            writer.writerow([row['response']])
    
    print(f"Successfully saved {len(filtered)} paragraphs")
    
    return output_file


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'together'],
        required=True,
        help='API provider to use (openai or together)'
    )
    parser.add_argument(
        '--keyword',
        type=str,
        required=True,
        help='Domain keyword for textbook generation'
    )
    parser.add_argument(
        '--stages',
        type=str,
        nargs='+',
        choices=['subfields', 'bulletpoints', 'chapters', 'postprocess', 'all'],
        default=['all'],
        help='Stages to execute'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--num-chapters-per-bp',
        type=int,
        default=5,
        help='Number of chapter samples per bullet point'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Model name to use'
    )
    parser.add_argument(
        '--num-subfields',
        type=int,
        default=10,
        help='Number of subfields to generate'
    )
    parser.add_argument(
        '--audiences',
        type=str,
        nargs='+',
        default=['elementary-school student', 'high-school student', 
                'undergraduate student', 'PhD student'],
        help='Target audiences for content generation'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=45,
        help='Minimum word count for post generation filtering'
    )
    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Show statistics during post-processing'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for text generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=8192,
        help='Maximum tokens for text generation'
    )
    
    args = parser.parse_args()
    
    if 'all' in args.stages:
        stages = ['subfields', 'bulletpoints', 'chapters', 'postprocess']
    else:
        stages = args.stages
    
    if args.provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        model_name = args.model_name
    else:
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        model_name = args.model_name 
    
    normalized_model = model_name.split('/')[-1]
    
    print(f"\n{'='*60}")
    print(f"TEXTBOOK GENERATION PIPELINE")
    print(f"{'='*60}")
    print(f"Keyword: {args.keyword}")
    print(f"Provider: {args.provider}")
    print(f"Model: {model_name}")
    print(f"Normalized Model: {normalized_model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Stages: {', '.join(stages)}")
    print(f"{'='*60}")
    
    client = get_client(args.provider, api_key)
    os.makedirs(args.data_path, exist_ok=True)
    
    if 'subfields' in stages:
        subfields = generate_subfields(
                        client, 
                        model_name, 
                        args.provider, 
                        args.keyword, 
                        args.num_subfields,
                        args.temperature, 
                        args.max_tokens
                    )
        
        subfields_file = f"{args.data_path}/{args.keyword}_{normalized_model}_subfields.json"
        with open(subfields_file, 'w') as f:
            json.dump({'keyword': args.keyword, 'model': model_name, 'subfields': subfields}, f, indent=4)
        print(f"\nSubfields saved to {subfields_file}")
    else:
        subfields_file = f"{args.data_path}/{args.keyword}_{normalized_model}_subfields.json"
        if os.path.exists(subfields_file):
            with open(subfields_file, 'r') as f:
                data = json.load(f)
                subfields = data['subfields']
        else:
            print(f"Warning: {subfields_file} not found. Run with --stages subfields first.")
            return
    
    if 'bulletpoints' in stages:
        generate_bullet_points(
            client, 
            model_name, 
            args.provider,
            args.keyword, 
            subfields, 
            args.audiences, 
            args.data_path, 
            20, 
            args.temperature, 
            args.max_tokens
        )
    
    chapters_file = None
    if 'chapters' in stages:
        chapters_file = generate_chapters(
                            client, 
                            model_name, 
                            args.provider,
                            args.keyword, 
                            args.data_path, 
                            args.num_chapters_per_bp,
                            args.temperature, 
                            args.max_tokens
                        )
    
    if 'postprocess' in stages:
        if not chapters_file:
            chapters_file = f'{args.data_path}/{args.keyword}_{normalized_model}_textbook_generated.json'
        
        if os.path.exists(chapters_file):
            output_csv = f'{args.data_path}/{args.keyword}_{normalized_model}_textbook_processed.csv'
            postprocess_chapters(
                chapters_file, output_csv, 
                args.threshold, args.show_stats
            )
        else:
            print(f"Warning: {chapters_file} not found. Run with --stages chapters first.")
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()