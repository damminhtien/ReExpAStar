
import argparse
from reexpastar.wa_star_cr import compare_on_grid  # type: ignore[attr-defined]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=50)
    p.add_argument('--height', type=int, default=50)
    p.add_argument('--weight', type=float, default=1.5)
    p.add_argument('--r_cr', type=float, default=0.2)
    args = p.parse_args()
    rows = compare_on_grid(args.width, args.height, args.weight, args.r_cr)
    keys = ['algo','w','r','cost','expansions','reopens','generated','runtime_ms','path_len']
    print(','.join(keys))
    for row in rows:
        print(','.join(str(row[k]) for k in keys))

if __name__ == "__main__":
    main()
