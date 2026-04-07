from qdrant_client import QdrantClient

def main():
    try:
        client = QdrantClient(
            host="172.18.216.71",
            port=6333,
            timeout=10
        )

        collections = client.get_collections()
        print("Qdrant 连接成功！")
        print(collections)

    except Exception as e:
        print("Qdrant 连接失败：", str(e))

if __name__ == "__main__":
    main()