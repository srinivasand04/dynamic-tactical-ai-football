# FootballBERT player vectors in the graph model: 15 World Cup matches for training, 1 held out, 3 folds x 3 conditions
cd "$(dirname "$0")/../.."
for h in 10504 10506 10513; do
  for c in off on shuffle; do
    f=outputs/fbert_gnn/${c}_${h}.log
    [ -s "$f" ] && grep -q "held-out real" "$f" && continue
    extra=""; [ "$c" != off ] && extra="--player-emb $c"
    .venv/Scripts/python.exe -u scripts/train_gnn.py --only-extra --extra-train pff:16 --holdout $h --graphs delaunay --no-sims \
      --tag fbert_${c}_${h} $extra > "$f" 2>&1
    echo "$c $h: $(grep 'held-out real' "$f")"
  done
done
