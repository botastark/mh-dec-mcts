# Pierdicca Fig. 4 from the new code: IG, r = 0..5 x weights equal / biased / adaptive
for w in equal biased adaptive; do for r in 0 1 2 3 4 5; do
  python scripts/run.py configs/pierdicca_fig4.yaml --planners ig --weights $w --r $r --out results/fig4
done; done
python scripts/plot_fig4.py results/fig4
