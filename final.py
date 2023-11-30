import numpy as np
import networkx as nx
from copy import deepcopy

# Our approach will be to find possible assignments using the Hungarian Algorithm
# How the Hungarian Algorithm works is explained in the Report


#In this function, we will be be finding the row which conatins the fewest number of zeroes.
#Then we will select the zero on the row and mark the corresponding row and column as false.
def least_zero_row(zero_matrix, marked_zero):
    row_sums = np.sum(zero_matrix == True, axis=1)
    rows_with_zeros = np.where(row_sums > 0)[0]

    if len(rows_with_zeros) > 0:
        least_row_idx = np.argmin(row_sums[rows_with_zeros])
        least_row = [row_sums[rows_with_zeros][least_row_idx], rows_with_zeros[least_row_idx]]

        zero_indices = np.where(zero_matrix[least_row[1]] == True)[0]
        zero_index = zero_indices[0]

        marked_zero.append((least_row[1], zero_index))
        zero_matrix[least_row[1], :] = False
        zero_matrix[:, zero_index] = False


# This function is responsible for finding the solutions to the Linear Assignment Problem
# It converts the given matrix into a boolean matrix where a value of 0 is represented as True and other values are represented as False.
# It then updates 'marked_rows' with rows that contain zeros and 'marked_columns' with columns that contain zeros.
# Identifies unmarked rows and checks for additional marked columns in those rows.
# Determines the final set of marked rows by subtracting the set of non-marked rows from the complete set of rows in the matrix.

def mark_matrix(matrix):
    current_matrix = matrix
    zero_bool_matrix = (current_matrix == 0)
    zero_bool_matrix_copy = zero_bool_matrix.copy()

    marked_zero = []
    while True in zero_bool_matrix_copy:
        least_zero_row(zero_bool_matrix_copy, marked_zero)

    marked_zero_row = [row for row, _ in marked_zero]
    marked_zero_column = [col for _, col in marked_zero]

    non_marked_row_set = set(range(current_matrix.shape[0])) - set(marked_zero_row)
    non_marked_row = list(non_marked_row_set)

    marked_columns_set = set()
    change = True
    while change:
        change = False
        for i in non_marked_row:
            row_array = zero_bool_matrix[i, :]
            marked_columns_set.update(set(np.where(row_array == True)[0]))

        marked_columns = list(marked_columns_set)
        
        for row_number, column_number in marked_zero:
            if row_number not in non_marked_row and column_number in marked_columns:
                non_marked_row.append(row_number)
                non_marked_row_set.add(row_number)
                change = True

    marked_rows = list(set(range(matrix.shape[0])) - non_marked_row_set)

    return marked_zero, marked_rows, marked_columns



# It first identifies the non-zero elements that haven't been covered or marked by rows or columns 
# It locates the minimum value among these non-zero elements
# Subtracts the minimum non-zero element found in the previous step from all uncovered elements. Adds the minimum value to the elements that are covered by both rows and columns
# Returns the modified matrix after adjusting the elements

def adjust_matrix(matrix, cover_rows, cover_columns):
    current_matrix = matrix
    non_zero_indices = np.ix_([i for i in range(len(current_matrix)) if i not in cover_rows],
                              [j for j in range(len(current_matrix[0])) if j not in cover_columns])
    
    non_zero_elements = current_matrix[non_zero_indices]
    min_number = np.min(non_zero_elements)

    current_matrix[non_zero_indices] -= min_number

    cover_indices = np.ix_(cover_rows, cover_columns)
    current_matrix[cover_indices] += min_number
    
    return current_matrix



# It starts by adjusting the input matrix for row and column reductions.
# Enters an iterative process to find the optimal assignment
# Continues iterations until all rows and columns are covered by the marked positions or until an optimal assignment is achieved.
# The function returns the positions of the elements that contribute to the optimal assignment

def hungarian_algorithm(matrix):
    dim = matrix.shape[0]
    current_matrix = matrix.copy()

    current_matrix -= np.min(current_matrix, axis=1)[:, np.newaxis]
    current_matrix -= np.min(current_matrix, axis=0)

    zero_count = 0
    while zero_count < dim:
        ans_position, marked_rows, marked_columns = mark_matrix(current_matrix)
        zero_count = len(marked_rows) + len(marked_columns)

        if zero_count < dim:
            current_matrix = adjust_matrix(current_matrix, marked_rows, marked_columns)

    return ans_position



# It calculates the total cost of the assignment by summing up the costs of the elements assigned according to the positions obtained from the Hungarian Algorithm.

def ans_calculation(matrix, pos):
	total = 0
	ans_matrix = np.zeros((matrix.shape[0],matrix.shape[1]))
	for i in range(len(pos)):
		total += matrix[pos[i][0],pos[i][1]]
		ans_matrix[pos[i][0], pos[i][1]] = matrix[pos[i][0], pos[i][1]]
	return total,ans_matrix


# Returns the matrix representing the assigned elements that contribute to the optimal solution.
def solve(Matrix):
	cost_matrix = Matrix
	ans_position = hungarian_algorithm(cost_matrix.copy()) 
	ans, ans_matrix = ans_calculation(cost_matrix,ans_position) 
	return [int(ans),ans_matrix]

Professor_Dictionary = {}
Professors = {}
Error ='No assignment possible'
Error2 ='Unequal courses and professors'
N = 0
Course_Dictionary_1 = {}
Course_Dictionary_2 = {}
course_count = 0
professor_count = 0

with open("input.txt") as Data_In:
    Course_List = Data_In.readline().split()
    course_count = len(Course_List)
    L = course_count*2
    Count = 0
    for i in range(L):
        Course_Dictionary_1[i] = Course_List[i // 2]
        Course_Dictionary_2[Course_List[i // 2]] = [i - (i % 2), i - (i % 2) + 1]
    Hungarian_Matrix = np.ones(shape=(L,L)) *1000
    while 1:
        Professor = Data_In.readline().split()
        if Professor:
            professor_count += 1
            Count += 1
            Name, Type = Professor[0],float(Professor[1])
            Professors[Name] = [Type, set()]
            x = int(Type // 0.5)
            for i in range(x):
                Professor_Dictionary[N + i] = Name
            Courses_Professor = Professor[2:]
            for i in range(len(Courses_Professor)):
                Hungarian_Matrix[Course_Dictionary_2[Courses_Professor[i]][0]][N : N + x] = i + 1
                Hungarian_Matrix[Course_Dictionary_2[Courses_Professor[i]][1]][N : N + x] = i + 1
            N += x
        else:
            break

if N != L:
    with open("output.txt", "w") as output_file:
        output_file.write(Error2)
else:
    with open("output.txt", "w") as output_file:
        output_file.write('List of courses: {}\n'.format(Course_List))
        output_file.write("\nNumber of courses: {}\n".format(course_count))
        output_file.write("Number of professors: {}\n".format(professor_count))

        M, Solution_1 = solve(Hungarian_Matrix)
        if M >= 1000:
            output_file.write('\n\n' + Error + '\n')
        else:
            output_file.write('\n\nPossible Allotments :\n')
            Answer_Set = set()
        def Solve(Solution):
                Answer = deepcopy(Professors)
                for i in range(N):
                    for j in range(N):
                        if Solution[i][j]:
                                Course = Course_Dictionary_1[i]
                                Answer[Professor_Dictionary[j]][1].add(Course)
                Answer_Set.add(str(Answer))
        Solve(Solution_1)
        for i in range(N):
                for j in range(N):
                        original = Hungarian_Matrix[i][j]
                        Hungarian_Matrix[i][j] = 1000
                        m, Solution = solve(Hungarian_Matrix)
                        if m == M:
                                Solve(Solution)
                        Hungarian_Matrix[i][j] = original
        for Answer in Answer_Set:
                output_file.write(Answer + '\n')

