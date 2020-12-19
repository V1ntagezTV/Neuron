def _for(count, sum):
    num = int(input())
    print(num)
    if (num == 0):
        print("end!")
        print("count:"+str(count)+" sum"+str(sum))
        return 0
    count += 1
    sum += num
    _for(count, sum)

_for(0, 0)