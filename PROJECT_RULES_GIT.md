## MobSC 跨设备（两台电脑）Git 使用规范

目的：规范在两台设备之间开发、提交与同步代码的步骤，减少冲突与历史污染。

### 1) 一次性环境配置（两台设备都执行）
```bash
# 设置用户名和邮箱
git config --global user.name "<你的名字>"
git config --global user.email "<你的邮箱>"

# 推荐默认行为：拉取使用 rebase、fetch 自动清理失效远端分支
git config --global pull.rebase true
git config --global fetch.prune true

# 可选：显示更丰富日志
git config --global format.pretty oneline
```

### 2) 每次切换设备时（先做这两步）
```bash
# 1. 刷新远端分支/标签引用并清理失效分支
git fetch origin --prune

# 2. 在当前工作分支上拉取最新提交（以 export-onnx 为例）
git pull --rebase origin export-onnx
```
说明：`fetch` 只更新引用不改动工作区；`pull --rebase` 会把本地未推送提交放到最新远端之上，历史更整洁。

### 3) 日常开发流程（任一设备）
```bash
# 确认当前分支（建议在 export-onnx 上开发，或按需切换）
git branch --show-current

# 开发、修改代码...

# 查看变更
git status | cat
git diff | cat

# 分次或一次性提交
git add -A
git commit -m "<清晰的提交信息>"

# 推送到远端
git push
```

### 4) 另一个设备继续开发前
切换到另一台设备后，回到“2) 每次切换设备时”的两步：`git fetch origin --prune` + `git pull --rebase origin <branch>`。

### 5) 解决冲突规范
出现冲突时：
```bash
# 查看状态与冲突文件
git status | cat

# 打开文件逐一解决冲突（删除 <<<<<<<、=======、>>>>>>> 标记），按需要保留本地或远端版本

# 标记为已解决
git add <冲突文件...>

# 如果是在 rebase 中
git rebase --continue
# 如果是在 merge 中
git commit
```
特殊情况：
- 空补丁（No changes）→ `git rebase --skip`
- 放弃当前 rebase → `git rebase --abort`

### 6) 临时保存与回滚（可选）
```bash
# 临时搁置未提交改动
git stash push -m "wip"
# 取回
git stash pop

# 回退最近一次提交但保留工作区修改
git reset --soft HEAD~1

# 查看最近历史
git log --oneline -n 10
```

### 7) 常见问题
- 为什么会进入 detached HEAD？
  - rebase 过程中或 checkout 到具体提交/标签时属于正常现象。完成冲突解决后 `git rebase --continue` 即可。
- 远端删除了分支，本地仍有 `origin/xxx`？
  - 执行 `git fetch origin --prune` 清理。
- 两台设备有未推送提交时同步失败？
  - 在“来源设备”先 `git push`，再在“目标设备”执行“2) 每次切换设备时”的步骤。

### 8) 分支与提交规范
- 分支：以功能或阶段命名，例如 `export-onnx`、`feature/<name>`。
- 提交信息：简洁动词短语，必要时在正文补充动机与影响范围。
- 避免把大模型、数据、生成产物提交入库；相关文件已通过 `.gitignore` 管控。

### 9) 快速命令清单
```bash
# 同步并更新当前分支（常用）
git fetch origin --prune && git pull --rebase origin export-onnx

# 检查工作区与最近提交
git status | cat && git log --oneline -n 5

# 解决冲突后继续 rebase
git add <files> && git rebase --continue
```

—— 本文档适用于个人在两台设备之间协同同一 GitHub 仓库的场景。


