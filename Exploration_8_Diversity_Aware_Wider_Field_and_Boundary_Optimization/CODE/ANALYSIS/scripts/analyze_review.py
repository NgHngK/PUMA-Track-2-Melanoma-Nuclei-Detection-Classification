from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'COMMON' / 'src'))
import argparse, json
from puma_exploration8.review import read_review

p = argparse.ArgumentParser(description='Summarize optional blinded review')
p.add_argument('--input')
a = p.parse_args()
if not a.input:
    print(json.dumps({'status': 'NO_INPUT', 'analysis': 'analyze_review'}, indent=2))
    raise SystemExit(0)
df = read_review(a.input)
done = df['reviewer_label'].notna() & (df['reviewer_label'].astype(str).str.len() > 0)
print(
    json.dumps(
        {
            'rows': len(df),
            'reviewed': int(done.sum()),
            'uncertain': int(
                df.loc[done, 'uncertain'].astype(str).str.lower().isin(['1', 'true', 'yes']).sum()
            ),
        },
        indent=2,
    )
)
