#!/bin/bash
##############################################################################
# VMEvalKit - Fix All Task Issues
# 
# This script fixes naming mismatches and missing prompt.txt files to enable
# all 1,643 tasks to be discovered and used for inference.
##############################################################################

set -e

cd /home/hokindeng/VMEvalKit/data/questions

echo "🔧 Fixing task discovery issues..."
echo "=================================="
echo ""

# Fix 1: object_rearr_task - Rename folders
echo "📝 Fix 1: Renaming object_rearr_task folders..."
cd object_rearr_task
count=0
for dir in obj_rearrange_*; do
    if [ -d "$dir" ]; then
        new_name=$(echo "$dir" | sed 's/obj_rearrange/object_rearr/')
        mv "$dir" "$new_name"
        count=$((count + 1))
    fi
done
echo "   ✅ Renamed $count folders: obj_rearrange_* → object_rearr_*"
cd ..
echo ""

# Fix 2: simple_scenes_task - Rename folders AND extract prompts
echo "📝 Fix 2: Fixing simple_scenes_task..."
cd simple_scenes_task
count=0
for dir in simplescenes_*; do
    if [ -d "$dir" ]; then
        # Extract prompt from metadata.json to prompt.txt
        if [ -f "$dir/question_metadata.json" ]; then
            python3 -c "
import json
with open('$dir/question_metadata.json', 'r') as f:
    data = json.load(f)
    prompt = data.get('prompt', '')
    if prompt:
        with open('$dir/prompt.txt', 'w') as p:
            p.write(prompt)
        print('   Created prompt.txt for $dir')
"
        fi
        
        # Rename folder
        new_name=$(echo "$dir" | sed 's/simplescenes/simple_scenes/')
        mv "$dir" "$new_name"
        count=$((count + 1))
    fi
done
echo "   ✅ Fixed $count folders: extracted prompts + renamed to simple_scenes_*"
cd ..
echo ""

# Verification
echo "🔍 Verification:"
echo "================"
cd /home/hokindeng/VMEvalKit

python3 << 'PYEOF'
import os
from pathlib import Path

questions_dir = Path("data/questions")
total = 0
issues = []

for task_dir in sorted(questions_dir.glob("*_task")):
    if not task_dir.is_dir():
        continue
    
    domain = task_dir.name.replace("_task", "")
    
    # Count folders matching expected pattern
    matching = list(task_dir.glob(f"{domain}_*"))
    
    if matching:
        # Check sample for required files
        sample = matching[0]
        has_prompt = (sample / "prompt.txt").exists()
        has_first = (sample / "first_frame.png").exists()
        
        if has_prompt and has_first:
            print(f"✅ {task_dir.name:30} {len(matching):4} tasks")
            total += len(matching)
        else:
            missing = []
            if not has_prompt: missing.append("prompt.txt")
            if not has_first: missing.append("first_frame.png")
            print(f"❌ {task_dir.name:30} {len(matching):4} tasks (missing: {', '.join(missing)})")
            issues.append(task_dir.name)
    else:
        all_folders = list(task_dir.glob("*"))
        all_folders = [f for f in all_folders if f.is_dir()]
        print(f"⚠️  {task_dir.name:30} {len(all_folders):4} folders (naming mismatch)")
        issues.append(task_dir.name)

print(f"\n{'='*60}")
print(f"Total usable tasks: {total}")
if issues:
    print(f"Issues remaining: {', '.join(issues)}")
else:
    print("✅ All tasks fixed and ready for inference!")
print(f"{'='*60}")
PYEOF

echo ""
echo "✅ All fixes completed!"

