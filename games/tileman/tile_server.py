from envs.multi_agent_env import TileServer
import sys

# fuck this shit

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else 9909
    server = TileServer(port=port)
    server.start()