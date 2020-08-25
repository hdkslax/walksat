import random
import time


def CNF_generator(num_vbles, num_clauses, num_vble_per_line):
    vbles = list()
    for i in range(1, num_vbles+1):
        vbles.append(i)
    p = "p cnf " + str(num_vbles) + " " + str(num_clauses)
    clauses = list()
    clauses.append(p)
    random.seed(time.time())
    for i in range(num_clauses):
        literals = list()
        for j in range(num_vble_per_line):
            var = random.choice(vbles)
            operator = random.randint(0, 1)
            if operator == 0:
                var = -var
            literals.append(str(var))
        literals.append('0')
        print(literals)
        clauses.append(literals)
    # print(clauses)

    return clauses


def write_CNF_file(clauses):
    title = "test_files/" + str(len(clauses)-1) + "-CNF.txt"
    f = open(title, 'w')
    c = "c " + str(len(clauses)-1) + "-CNF problem"
    f.write(c)
    f.write("\n")
    for j in range(len(clauses[0])):
        f.write(clauses[0][j])
    f.write("\n")
    for i in range(1, len(clauses)):
        for j in range(len(clauses[i])):
            f.write(clauses[i][j])
            if j != len(clauses[i])-1:
                f.write(" ")
        f.write("\n")


    f.close()


def main():
    clauses = CNF_generator(1000, 3000, 3)
    write_CNF_file(clauses)

    clauses = CNF_generator(5000, 5000, 2)
    write_CNF_file(clauses)


if __name__ == '__main__':
    main()
