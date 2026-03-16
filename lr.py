import socket, json, os, random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

MODEL_FILE = "ci_model.pth"
COMMAND_MAP = {0: 'N', 1: 'U', 2: 'D', 3: 'L', 4: 'R', 5: 'S'}
INPUT_SIZE, OUTPUT_SIZE = 80, 6

class DQN(nn.Module):
    def __init__(self, in_s, out_s):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_s, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, out_s))
    def forward(self, x): return self.fc(x)

class GameAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(INPUT_SIZE, OUTPUT_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.epsilon, self.epsilon_min, self.epsilon_decay = 1.0, 0.1, 0.998
        self.gamma, self.batch_size, self.train_step = 0.95, 32, 0
        if os.path.exists(MODEL_FILE):
            checkpoint = torch.load(MODEL_FILE); self.model.load_state_dict(checkpoint['model_state'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
    def save_model(self): torch.save({'model_state': self.model.state_dict(), 'epsilon': self.epsilon}, MODEL_FILE)
    def get_action(self, state):
        if random.random() <= self.epsilon: return random.randrange(OUTPUT_SIZE)
        st = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)
        with torch.no_grad(): return torch.argmax(self.model(st)).item()
    def train(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s)).view(self.batch_size, -1).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).view(self.batch_size, -1).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
        curr_q = self.model(s).gather(1, a); next_q = self.model(ns).max(1)[0].detach()
        target = r + (self.gamma * next_q * (1 - d))
        loss = nn.MSELoss()(curr_q.squeeze(), target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
        self.train_step += 1

agent = GameAgent()

def socket_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1', 5001)); s.listen(1)
    conn, addr = s.accept()
    
    health, last_grid, last_action, buffer, ship_missing = 5, None, None, "", False

    while True:
        try:
            data = conn.recv(16384).decode('utf-8')
            if not data: break
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                p = json.loads(line)
                grid = np.zeros((8, 10))
                ship_pos = None
                for d in p.get('d', []):
                    x, y = max(0, min(9, d['x'])), max(0, min(7, d['y']))
                    grid[y][x] = {9:1, 3:-1, 4:-2, 8:-1, 0:2, 1:2}.get(d['c'], 0)
                    if d['c'] == 9: ship_pos = (x, y)

                reward, done = 0.1, False
                if not ship_pos:
                    if not ship_missing:
                        reward, health, done, ship_missing = -20.0, health-1, True, True
                        if health <= 0: health = 5; agent.save_model() # RESET
                else: ship_missing = False

                if last_grid is not None:
                    agent.memory.append((last_grid, last_action, reward, grid, done)); agent.train()

                action_idx = agent.get_action(grid)
                cmd = COMMAND_MAP[action_idx]
                
                # Sınır Kontrolü Cezası
                if ship_pos:
                    if (ship_pos[0] == 0 and cmd == 'L') or (ship_pos[0] == 9 and cmd == 'R') or \
                       (ship_pos[1] == 0 and cmd == 'U') or (ship_pos[1] == 7 and cmd == 'D'):
                        reward = -1.0

                conn.send(cmd.encode())
                last_action, last_grid = action_idx, grid

                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"CAN: {'❤ ' * health} | EPS: {agent.epsilon:.3f} | STEP: {agent.train_step}")
                print(f"AKSİYON: {cmd} | REWARD: {reward:.2f}")
                grid_chars = {1:"▲", -1:"O", -2:"°", 2:"$"}
                print("+" + "---"*10 + "+")
                for row in grid: print("|" + "".join(f" {grid_chars.get(v, '.')} " for v in row) + "|")
                print("+" + "---"*10 + "+")
                if agent.train_step % 200 == 0: agent.save_model()
        except: break
    conn.close()

if __name__ == "__main__": socket_server()