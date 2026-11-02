cd /c/Users/Srinivasan/grf-prototype
while ! grep -q "^converted" outputs/convert_pff_log.txt; do sleep 20; done
echo "== conversion finished"; grep "^converted" outputs/convert_pff_log.txt
for g in $(grep -o "'match_id': '[0-9]*', 'error'" outputs/convert_pff_log.txt | grep -o "[0-9]\{4,5\}"); do echo "== redo $g"; python scripts/convert_pff.py $g; done
echo "== step 2: key states + graphs"; python scripts/run_pipeline.py --dataset pff
echo "== step 3: quality"; python scripts/data_quality.py --dataset dfl pff
echo "== ALL DONE"
