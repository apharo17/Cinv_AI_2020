
N = 8

def bfs_queens(s, t):
    pass



class Node:

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
            v = Node()
            v.fromlist(self.col_to_row, self.col+1)
            neighbors.append(v)

        self.col_to_row[self.col+1] = -1

        return neighbors


    def print_node(self):
        print(self.col_to_row)

s = Node()
s.fromrow(0)
for u in s.get_neighbors():
    for v in u.get_neighbors():
        v.print_node()
    print()
