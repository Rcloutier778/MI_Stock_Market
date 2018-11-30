

def main():
    with open("output.txt", 'r') as f:
        lines = f.readlines()
        stage=0
        data={}
        tline=None
        std=None
        for line in lines:
            line=line.strip()
            if '###' in line:
                stage+=1
                continue
            if 'lbfgs' in line or 'sgd' in line or 'adam' in line:
                tline=line.strip()
                continue
            if tline:
                if std==1:
                    data[tline][stage+3]=float(line)
                    tline=None
                    std=None
                    continue
                if not stage:
                    data[tline] = [float(line.split(': ')[-1]),0,0,0,0]
                    std = 1
                else:
                    data[tline][1]=float(line.split(': ')[-1])
                    data[tline][2]=((float(data[tline][1]) - float(data[tline][0]))/data[tline][0])*100
                    std = 1
                
                    
    sorted_data = sorted(data.items(), key=lambda kv: kv[1][2])
    for i in sorted_data:
        print("%28s:  %3.3f  %2.2f  %2.2f  %0.3f  %0.3f" % (i[0], i[1][2], i[1][0], i[1][1], i[1][3], i[1][4]))