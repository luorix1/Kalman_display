file = open("data_xy.txt", "r")
new = open("data_rt.txt", "w")

line = file.readline()
while line:
    new.write(line)
    line = file.readline()

file.close()
new.close()