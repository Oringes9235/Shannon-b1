# PARADIGM

## Git Commit 信息规范范式

### 整体格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### 1. Header（头部，必填）
一行内容，格式为：`<type>(<scope>): <subject>`

| 字段 | 必填 | 描述 |
|------|------|------|
| `type` | ✅ | 提交类型，见下方类型表 |
| `scope` | ❌ | 影响范围，如模块名或功能名 |
| `subject` | ✅ | 简短描述，5-10个字说明本次提交做了什么 |

#### type 可选值
| type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `docs` | 文档修改 |
| `style` | 代码格式修改（不影响功能） |
| `refactor` | 重构 |
| `perf` | 性能提升 |
| `test` | 测试相关 |
| `build` | 构建系统 |
| `ci` | CI 配置文件修改 |
| `chore` | 构建流程、依赖库或工具的变动 |
| `revert` | 回滚版本 |

#### Header 示例
```
feat(user): 用户模块加入短信登录功能

fix(cart): 修复购物车数量计算错误

docs: 更新部署文档
```

---

### 2. Body（正文，可选）
对本次提交的**详细描述**，解释：
- **为什么**做这个改动
- **解决了什么问题**
- 与之前实现方式的差异等

#### Body 示例
```
因分布式锁超时设置不当导致重复执行，现调整锁超时时间为5秒，并增加重试机制。
```

---

### 3. Footer（脚注，可选）
包含两类信息：

| 字段 | 用途 |
|------|------|
| `Breaking Changes` | 说明不兼容的变动及迁移方法（不常用） |
| `Closed Issues` | 关闭的 issue 或缺陷编号 |

#### Footer 示例
```
Breaking Changes: 升级到 v2 API，移除了旧的 /v1/login 接口，请改用 /v2/auth。

Closed Issues: Closes #234, #456
```

---

## 完整提交示例

```
feat(order): 订单模块加入超时自动取消功能

为解决用户下单后长时间未支付导致的库存占用问题，增加定时任务扫描超时订单并自动取消。

Closed Issues: Closes #128
```

使用时，你可以按需省略 `scope`、`body` 或 `footer`，但 `type` 和 `subject` 必须保留。