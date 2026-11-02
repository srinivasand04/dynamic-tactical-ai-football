cd /c/Users/Srinivasan/grf-prototype
until grep -q "^finished in" outputs/team_policy_log.txt && [ -f outputs/team_policy/pff_decisions.json ] && grep -q "^wrote outputs/team_policy/pff_decisions.json" outputs/team_decisions_log.txt; do sleep 30; done
echo "== $(date) models ready, league starts"
.venv/Scripts/python.exe scripts/team_league.py --teams Argentina France Croatia --games 6 --ai-games 4 --procs 5 > outputs/team_league_log.txt 2>&1
echo "== $(date) league done; recording Argentina v France"
.venv/Scripts/python.exe scripts/team_league.py --video Argentina France > outputs/team_league_video_log.txt 2>&1
ffmpeg -y -loglevel error -i outputs/team_league/Argentina_vs_France.mp4 -c:v libx264 -crf 26 -pix_fmt yuv420p outputs/team_league/Argentina_vs_France_h264.mp4
echo "== $(date) ALL DONE"
