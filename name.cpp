AI w1: toy prob (Water Jug)

#include <iostream>
#include <unordered_set>

using namespace std;

struct State {
    int x, y;

    bool operator==(const State& other) const {
        return x == other.x && y == other.y;
    }
};

namespace std {
    template <>
    struct hash<State> {
        size_t operator()(const State& s) const {
            return hash<int>()(s.x) ^ hash<int>()(s.y);
        }
    };
}

bool isGoalState(const State& state, int target) {
    return state.x == target || state.y == target;
}

void pourWater(int capacityX, int capacityY, int target, State currentState, unordered_set<State>& visited) {
    if (isGoalState(currentState, target)) {
        cout << "Target " << target << " reached! ("
             << currentState.x << ", " << currentState.y << ")" << endl;
        return;
    }

    visited.insert(currentState);

    // Pour water from jug X to jug Y
    if (currentState.x > 0) {
        int pourAmount = min(currentState.x, capacityY - currentState.y);
        State nextState = {currentState.x - pourAmount, currentState.y + pourAmount};

        if (visited.find(nextState) == visited.end()) {
            cout << "Pour " << pourAmount << " from X to Y (" << nextState.x << ", " << nextState.y << ")" << endl;
            pourWater(capacityX, capacityY, target, nextState, visited);
        }
    }

    // Pour water from jug Y to jug X
    if (currentState.y > 0) {
        int pourAmount = min(currentState.y, capacityX - currentState.x);
        State nextState = {currentState.x + pourAmount, currentState.y - pourAmount};

        if (visited.find(nextState) == visited.end()) {
            cout << "Pour " << pourAmount << " from Y to X (" << nextState.x << ", " << nextState.y << ")" << endl;
            pourWater(capacityX, capacityY, target, nextState, visited);
        }
    }

    // Empty jug X
    if (currentState.x > 0) {
        State nextState = {0, currentState.y};

        if (visited.find(nextState) == visited.end()) {
            cout << "Empty X (" << nextState.x << ", " << nextState.y << ")" << endl;
            pourWater(capacityX, capacityY, target, nextState, visited);
        }
    }

    // Empty jug Y
    if (currentState.y > 0) {
        State nextState = {currentState.x, 0};

        if (visited.find(nextState) == visited.end()) {
            cout << "Empty Y (" << nextState.x << ", " << nextState.y << ")" << endl;
            pourWater(capacityX, capacityY, target, nextState, visited);
        }
    }

    // Fill jug X
    if (currentState.x < capacityX) {
        State nextState = {capacityX, currentState.y};

        if (visited.find(nextState) == visited.end()) {
            cout << "Fill X (" << nextState.x << ", " << nextState.y << ")" << endl;
            pourWater(capacityX, capacityY, target, nextState, visited);
        }
    }

    // Fill jug Y
    if (currentState.y < capacityY) {
        State nextState = {currentState.x, capacityY};

        if (visited.find(nextState) == visited.end()) {
            cout << "Fill Y (" << nextState.x << ", " << nextState.y << ")" << endl;
            pourWater(capacityX, capacityY, target, nextState, visited);
        }
    }
}

int main() {
    int capacityX, capacityY, target;

    cout << "Enter capacity of jug X: ";
    cin >> capacityX;

    cout << "Enter capacity of jug Y: ";
    cin >> capacityY;

    cout << "Enter target amount: ";
    cin >> target;

    State initialState = {0, 0};
    unordered_set<State> visited;

    cout << "Solution steps:" << endl;
    pourWater(capacityX, capacityY, target, initialState, visited);

    return 0;
}


AI w1: Camel Banana

total=int(input('Enter no. of bananas at starting '))
distance=int(input('Enter distance you want to cover '))
load_capacity=int(input('Enter max load capacity of your camel '))
lose=0
start=total
for i in range(distance):
    while start>0:
        start=start-load_capacity
        if start==1:
            lose=lose-1
        lose=lose+2
    lose=lose-1
    start=total-lose
    if start==0:
        break
print(start)

AI w2: Graph Coloring

colors = ['red','blue','green','orange','yellow','violet']

states = ['MP','New Delhi','Haryana','Rajasthan','Gujarat']

neighbours = {
    'MP':['New Delhi','Rajasthan','Gujarat'],
    'New Delhi':['MP','Rajasthan','Haryana'],
    'Haryana':['New Delhi'],
    'Rajasthan':['MP','Gujarat','New Delhi'],
    'Gujarat':['Rajasthan','MP']
}

state_colors = {}
def promising(state, color):
    for neighbour in neighbours.get(state):
        color_of_neighbor = state_colors.get(neighbour)
        if color_of_neighbor == color:
            return False

    return True
   
for state in states:
    for color in colors:
        if promising(state, color):
            state_colors[state] = color

print (state_colors)

AI w3: Constraint Satisfaction Prob (CSP)

def solutions():
    # letters = ('s', 'e', 'n', 'd', 'm', 'o', 'r', 'y')
    all_solutions = list()
    for s in range(9, -1, -1):
        for e in range(9, -1, -1):
            for n in range(9, -1, -1):
                for d in range(9, -1, -1):
                    for m in range(9, 0, -1):
                        for o in range(9, -1, -1):
                            for r in range(9, -1, -1):
                                for y in range(9, -1, -1):
                                    if len(set([s, e, n, d, m, o, r, y])) == 8:
                                        send = 1000 * s + 100 * e + 10 * n + d
                                        more = 1000 * m + 100 * o + 10 * r + e
                                        money = 10000 * m + 1000 * o + 100 * n + 10 * e + y

                                        if send + more == money:
                                            all_solutions.append(
                                                (send, more, money))
    return all_solutions


print(solutions())


AI w4-w5: BFS-DFS

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
visited_bfs = []
queue = []


def bfs(visited_bfs, graph, node):
  visited_bfs.append(node)
  queue.append(node)

  while queue:
    s = queue.pop(0)
    print(s, end=" ")

    for neighbour in graph[s]:
      if neighbour not in visited_bfs:
        visited_bfs.append(neighbour)
        queue.append(neighbour)


visited = set()


def dfs(visited, graph, node):
    if node not in visited:
        print(node, end=" ")
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)


print("BFS:", end=" ")
bfs(visited_bfs, graph, 'A')
print('\n')
print("DFS:", end=" ")
dfs(visited, graph, 'A')

print(test[0])


AI w6: A* algorithm

from queue import PriorityQueue


class Graph:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    def h(self, n):
        H = {
            'A': 1,
            'B': 1,
            'C': 1,
            'D': 1
        }

        return H[n]

    def best_first_search(self, start, goal):
        explored = []
        pq = PriorityQueue()
        pq.put((0, start))
        parents = {start: None}

        while not pq.empty():
            current = pq.get()[1]

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parents[current]
                path.reverse()
                print(f"Best-First Search path: {path}")
                return path

            explored.append(current)

            for neighbor, weight in self.get_neighbors(current):
                if neighbor not in explored and neighbor not in [i[1] for i in pq.queue]:
                    parents[neighbor] = current
                    pq.put((self.h(neighbor), neighbor))

        print("Path not found!")
        return None

    def a_star_algorithm(self, start_node, stop_node):
        open_list = set([start_node])
        closed_list = set([])
        g = {}

        g[start_node] = 0
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v

            if n == None:
                print('Path does not exist!')
                return None
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('A* path: {}'.format(reconst_path))
                return reconst_path

            for (m, weight) in self.get_neighbors(n):
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None


adjacency_list = {
    'A': [('B', 1), ('C', 3), ('D', 7)],
    'B': [('D', 5)],
    'C': [('D', 12)]
}
graph1 = Graph(adjacency_list)
graph1.best_first_search('A', 'D')
graph1.a_star_algorithm('A', 'D')

AI w7: Alpha-Beta Pruning

MAX, MIN = 1000, -1000
def minimax(depth, nodeIndex, maximizingPlayer,
values, alpha, beta):

if depth == 3:
return values[nodeIndex]

if maximizingPlayer:

best = MIN

for i in range(0, 2):

val = minimax(depth + 1, nodeIndex * 2 + i,
False, values, alpha, beta)
best = max(best, val)
alpha = max(alpha, best)

if beta <= alpha:
break

return best

else:
best = MAX
for i in range(0, 2):

val = minimax(depth + 1, nodeIndex * 2 + i,
True, values, alpha, beta)
best = min(best, val)
beta = min(beta, best)
if beta <= alpha:
break

return best

if __name__ == "__main__":

    values = []
    for i in range(0, 8):

        x = int(input(f"Enter Value {i}  : "))
        values.append(x)

    print ("The optimal value is :", minimax(0, 0, True, values, MIN, MAX))


AI w9: Monty Hall

# Monty Hall Game in Python
import random

def play_monty_hall(choice):
    # Prizes behind the door
    # initial ordering doesn't matter
    prizes = ['goat', 'car', 'goat']
   
    # Randomizing the prizes
    random.shuffle(prizes)
   
    # Determining door without car to open
    while True:
        opening_door = random.randrange(len(prizes))
        if prizes[opening_door] != 'car' and choice-1 != opening_door:
            break
   
    opening_door = opening_door + 1
    print('We are opening the door number-%d' % (opening_door))
   
    # Determining switching door
    options = [1,2,3]
    options.remove(choice)
    options.remove(opening_door)
    switching_door = options[0]
        # Asking for switching the option
    print('Now, do you want to switch to door number-%d? (yes/no)' %(switching_door))
    answer = input()
    if answer == 'yes':
        result = switching_door - 1
    else:
        result = choice - 1
   
    # Displaying the player's prize
    print('And your prize is ....', prizes[result].upper())
   
# Reading initial choice
choice = int(input('Which door do you want to choose? (1,2,3): '))

# Playing game
play_monty_hall(choice)


AI w10: ML (Linear Reg)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generating some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Calculating Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


AI- NLP

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Sample text for analysis
text = "I really enjoyed this movie. The acting was great and the plot was engaging."

# Calculate the sentiment score for the text
score = sia.polarity_scores(text)

# Print the sentiment score
print("negative = ", score["neg"])
print("neutral = ", score["neu"])
print("positive = ", score["pos"])
print("compound = ", score["compound"])


AI- Fuzzy Logic

# Difference Between Two Fuzzy Sets for A_key in A:  X[A_key]= 1-A[A_key] print('Fuzzy Set Complement is :', X)
A = dict()
B = dict()
Y = dict()
X = dict()
A = {"a": 0.2, "b": 0.3, "c": 0.6, "d": 0.6}
B = {"a": 0.9, "b": 0.9, "c": 0.4, "d": 0.5}
print('The First Fuzzy Set is :', A)
print('The Second Fuzzy Set is :', B)
for A_key, B_key in zip(A, B):        
    A_value = A[A_key]        
    B_value = B[B_key]
    if A_value > B_value:                
        Y[A_key] = A_value
    else:                
            Y[B_key] = B_value

print('Fuzzy Set Union is :', Y)