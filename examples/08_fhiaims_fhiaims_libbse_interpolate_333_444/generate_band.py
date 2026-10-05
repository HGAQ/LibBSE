# print a output band series
# format: 
# output  band   0.00000  0.00000  0.00000   0.00000  0.00000  0.50000  3
length = 3
width = 3
height = 3
for i in range(length):
    for j in range(width):
        print("output  band", end=' ')
        print("{:8.5f} {:8.5f} {:8.5f}   ".format(i/length, j/width, 0), end=' ')
        print("{:8.5f} {:8.5f} {:8.5f}   ".format(i/length, j/width, 1.0 - 1/height), end=' ')
        print(height)
