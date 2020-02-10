
N = 8


class Node:

    def __init__(self):
        self.col = 0
        self.col_to_row = [-1 for col in range(N)]
        self.parent = None


    def fromrow(self, row):
        self.col_to_row[0] = row


    def fromlist(self, col_to_row, last_col):
        self.col = last_col
        for col in range(last_col+1):
            self.col_to_row[col] = col_to_row[col]


    def get_parent(self):
        return  self.parent


    def set_parent(self, parent):
        self.parent = parent


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


    def print_path(self):
        u = self
        u.print_info()
        while u.parent is not None:
            u = u.parent
            u.print_info()


def bfs_queens(s, t, verbose=False):

    res = False
    frontier = []
    frontier.append(s)
    u = None

    while len(frontier) > 0:
        u = frontier.pop(0)
        if u.equals(t):
            res = True
            break
        neighbors = u.get_neighbors()
        frontier.extend(neighbors)
        for v in neighbors:
            v.parent = u

    if verbose:
        u.print_path()

    return res



s = Node()
s.fromrow(0)
t = Node()
t.fromlist([0,6,4,7,1,3,5,2], 7)
#t.print_node()
print(bfs_queens(s, t, verbose=True))