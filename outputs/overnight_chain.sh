cd /c/Users/Srinivasan/grf-prototype
PYA=.venv-analysis/Scripts/python.exe
waitpid() { while powershell -NoProfile -Command "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) {exit 0} else {exit 1}"; do sleep 30; done; }
echo "== $(date) waiting for the running team-model jobs (pff 22276, skillcorner 19528)"
waitpid 22276; echo "== $(date) pff team models finished"
waitpid 19528; echo "== $(date) skillcorner team models finished"
echo "== $(date) quality table, all datasets";        python scripts/data_quality.py --dataset dfl pff skillcorner > outputs/data_quality_log.txt 2>&1
echo "== $(date) team shape, skillcorner clubs";      python scripts/team_style.py --dataset skillcorner --min-matches 3 > outputs/team_style_skillcorner_log.txt 2>&1
echo "== $(date) DFL held-out folds with Delaunay graphs"; $PYA scripts/run_folds.py > outputs/run_folds_delaunay_log.txt 2>&1
echo "== $(date) pass-edge proof, seed 0, Delaunay graphs"; $PYA scripts/run_pass_proof.py --seeds 0 > outputs/run_pass_proof_delaunay_log.txt 2>&1
echo "== $(date) ALL DONE"
