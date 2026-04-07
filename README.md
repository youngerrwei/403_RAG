### LAB 403 RAG 系统文件结构
#### 环境变量 
.env - 用于设置通用变量
#### 知识库入库逻辑 
ingest.py - 用于创建知识库 Embedding
#### Agent 检索逻辑 
rag_agent.py - 用于检索方案路由，RAG 召回与重排序，回答生成
#### 网页后端逻辑 
web_app.py - 用于与网页前端传输数据
#### 网页前端页面
- 登录页面 templates/login.html
- 对话页面 templates/index.html
