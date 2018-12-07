
from collections import defaultdict
def main():
    with open("SVM output.txt", 'r') as f:
        lines = f.readlines()
        stage=0
        data={}
        tline=None
        std=None
        prevLine=None
        tt=None
        for line in lines:
            line=line.strip()
            if '###' in line:
                stage+=1
                continue
            if 'accuracy' in line:
                tline=prevLine
            if tline:
                if tt==1:
                    data[tline]['Time'].append(float(line.split(" ")[-1]))
                    tt=None
                    tline = None
                    continue
                if std==1:
                    if 'Time' in line:
                        data[tline]['std']=[0,0]
                        data[tline]['Time'].append(float(line.split(" ")[-1]))
                        std=None
                        tt=None
                        tline=None
                        continue
                    else:
                        data[tline]['std'].append(float(line))
                        
                        std=None
                        tt=1
                        continue
                if not stage:
                    data[tline] = {'Accuracy':[],'std':[],'Time':[],'Diff':None,'percent':None}
                    data[tline]['Accuracy'].append(float(line.split(': ')[-1]))
                    std = 1
                else:
                    data[tline]['Accuracy'].append(float(line.split(': ')[-1]))
                    data[tline]['Diff']=data[tline]['Accuracy'][1] - data[tline]['Accuracy'][0]
                    data[tline]['percent'] = 100*(data[tline]['Accuracy'][1] - data[tline]['Accuracy'][0])/data[tline]['Accuracy'][0]
                    std = 1
            prevLine=line
                
                    
    sorted_data = sorted(data.items(), key=lambda kv: kv[1]['Diff'])
    with open('SVM formattedOutput.txt','w+') as f:
        titleStr="%28s   %5s  %6s  %4s  %4s  %6s  %6s  %6s  %6s\n" % ("Method","Diff","percent","Train","Test","STD_tr","STD_te","Time_tr","Time_te")
        f.write(titleStr)
        print(titleStr[:-2])
        for i in sorted_data:
            if i[1]['Diff'] >= 0 and i[1]['Accuracy'][0] >=51.0 and i[1]['Time'][0] < 50.0:
                writestr="%28s:  %3.3f  %2.2f  %2.2f  %6.2f  %6.2f  %6.2f  %6.2f  %2.2f\n" % (i[0], i[1]['Diff'],i[1]['percent'], i[1]['Accuracy'][0], i[1]['Accuracy'][1],i[1]['std'][0], i[1]['std'][1],i[1]['Time'][0], i[1]['Time'][1], )
                f.write(writestr)
                print(writestr[:-2])
