
# 3. the figure from the new code
for w in equal biased adaptive; do for r in 0 1 2 3 4 5; do
  python scripts/run_single.py configs/pierdicca_fig4.yaml --planners ig --weights $w --r $r
done; done
python scripts/plot_paper_fig4.py results/single    

