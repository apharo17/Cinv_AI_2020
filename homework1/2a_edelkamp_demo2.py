
N = 8

class Vertex:

    def __init__(self):
        self.col = 0
        self.col_to_row = [-1 for col in range(N)]


    def fromrow(self, row):
        self.col_to_row[0] = row


    def fromlist(self, col_to_row, last_col):
        self.col = last_col
        for col in range(last_col+1):
            self.col_to_row[col] = col_to_row[col]


    def __get_next_rows(self, last_col):

        next_col = last_col+1
        if next_col >= N:
            return []

        flags = [True for i in range(N)]
        for col in range(next_col):
            row = self.col_to_row[col]
            flags[row] = False

            if row - (next_col - col) >= 0:
                flags[row - (next_col - col)] = False

            if row + (next_col - col) < N:
                flags[row + (next_col - col)] = False

        next_rows = []
        for row in range(N):
            if flags[row]:
                next_rows.append(row)

        return next_rows


    def get_neighbors(self):

        neighbors = []
        next_rows = self.__get_next_rows(self.col)
        for row in next_rows:
            self.col_to_row[self.col + 1] = row
            v = Vertex()
            v.fromlist(self.col_to_row, self.col+1)
            neighbors.append(v)

        if len(next_rows) > 0:
            self.col_to_row[self.col+1] = -1

        return neighbors


    def equals(self, t):
        for col in range(N):
            if self.col_to_row[col] != t.col_to_row[col]:
                return False
        return True


    def print_info(self):
        print(self.col_to_row)


def bfs_queens(s, t):
    frontier = []
    frontier.append(s)

    while len(frontier) > 0:
        u = frontier.pop(0)
        if u.equals(t):
            return True
        frontier.extend(u.get_neighbors())

    return False


'''s = Node()
s.fromrow(0)
for u in s.get_neighbors():
    for v in u.get_neighbors():
        v.print_node()
    print()'''


s = Vertex()
s.fromrow(0)
t = Vertex()
t.fromlist([0,6,4,7,1,3,5,2], 7)
#t.print_node()
print(bfs_queens(s, t))