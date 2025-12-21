# MolX Client

> Web UI Client for MolX Agent

MolX Agent 的 Web 客户端，提供现代化的聊天界面与 AI Agent 交互。

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| [Vite](https://vite.dev/) | ^6.0 | 构建工具 & 开发服务器 |
| [React](https://react.dev/) | ^18.3 | UI 框架 |
| [TypeScript](https://www.typescriptlang.org/) | ^5.6 | 类型安全 |
| [TailwindCSS](https://tailwindcss.com/) | ^3.4 | 样式系统 |
| [Shadcn UI](https://ui.shadcn.com/) | latest | 组件库 |
| [Tanstack Query](https://tanstack.com/query) | ^5.0 | 数据获取 & 缓存 |
| [Tanstack Router](https://tanstack.com/router) | ^1.0 | 路由管理 |
| [Vercel AI SDK](https://ai-sdk.dev/) | ^4.0 | 流式 AI 接口 |
| [Prettier](https://prettier.io/) | ^3.4 | 代码格式化 |

## 参考项目

- [Open Canvas](https://github.com/langchain-ai/open-canvas) - LangChain 的 Canvas UI
- [Gemini Chatbot](https://github.com/vercel-labs/gemini-chatbot) - Vercel AI SDK 示例

## 页面设计

```
┌─────────────────────────────────────────────────────────┐
│  MolX Agent                              [Settings]     │
├─────────────────┬───────────────────────────────────────┤
│  Chat List      │  Chat with Agent                      │
│  ┌───────────┐  │  ┌─────────────────────────────────┐  │
│  │ Session 1 │  │  │                                 │  │
│  │ Session 2 │  │  │   Chat Messages                 │  │
│  │ Session 3 │◀─│──│   (Markdown + Code)             │  │
│  │ ...       │  │  │                                 │  │
│  └───────────┘  │  ├─────────────────────────────────┤  │
│                 │  │ [Input message...]        [Send]│  │
│  [+ New Chat]   │  └─────────────────────────────────┘  │
└─────────────────┴───────────────────────────────────────┘
```

**功能**:
- **Chat List**: 会话列表管理 (创建、切换、删除)
- **Chat Panel**: 消息显示与流式响应
- **Markdown 渲染**: 支持代码高亮
- **响应式布局**: 适配桌面和移动端

## 目录结构

```
molx_client/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.ts
├── components.json          # Shadcn UI 配置
├── src/
│   ├── main.tsx             # 入口
│   ├── App.tsx              # 根组件
│   ├── index.css            # 全局样式
│   ├── lib/
│   │   ├── utils.ts         # 工具函数
│   │   └── api.ts           # API 客户端
│   ├── hooks/
│   │   ├── use-chat.ts      # AI SDK hook
│   │   └── use-sessions.ts  # 会话管理
│   ├── components/
│   │   ├── ui/              # Shadcn UI 组件
│   │   ├── chat/
│   │   │   ├── chat-list.tsx
│   │   │   ├── chat-panel.tsx
│   │   │   ├── message-item.tsx
│   │   │   └── input-form.tsx
│   │   └── layout/
│   │       ├── header.tsx
│   │       └── sidebar.tsx
│   ├── pages/
│   │   └── chat.tsx         # 主聊天页
│   └── types/
│       └── index.ts         # 类型定义
└── README.md
```

## 快速开始

### 安装依赖

```bash
cd molx_client
npm install
```

### 开发模式

```bash
npm run dev
# 访问 http://localhost:5173
```

### 生产构建

```bash
npm run build
npm run preview
```

## 配置

### 环境变量

```bash
# .env.local
VITE_API_BASE_URL=http://localhost:8000
```

### API 代理 (开发模式)

Vite 配置中已设置代理，将 `/api` 请求转发到 MolX Server:

```ts
// vite.config.ts
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

## 与 MolX Server 集成

确保 MolX Server 正在运行:

```bash
# 在 molx_agent 目录
make serve
# 或
uv run molx-server run
```

然后启动客户端开发服务器连接 API。