# Public repository checklist

```bash
cd AI_Shape_Sorting_Robot_Arm_Ultra
git init
git add .
git commit -m "Ultra adaptive shape sorting robot"
git branch -M main
git remote add origin <PUBLIC_REPOSITORY_URL>
git push -u origin main
```

Before publishing:

1. Run `grep -RniE 'email|token|password|secret' .` and inspect every match.
2. Confirm `python -m pytest -q` and the offline E2E pass.
3. Confirm the report PDF renders and the H.264 video plays.
4. Keep `out/genesis/qualification.json`; do not delete the failed qualification or claim Genesis execution.
5. After a real Genesis run, add the new run registry, environment snapshot, logs and video without overwriting surrogate results.
6. Ensure the repository visibility is public if required by the course submission rules.
