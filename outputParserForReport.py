import csv
s="""AMD (AMD)   59.0823   49.6198   -1.720
Verizon (VZ)   59.4151   50.8555   -0.836
J.P. Morgan (JPM)   63.4570   50.8555   -0.514
Paypal (PYPL)   69.0828   50.8876   0.457
Exxon Mobil (XOM)   62.4108   51.1407   1.413
Bank of America (BAC)   61.3172   51.9962   -0.571
Google (GOOGL)   97.8768   52.2949   -2.218
Macys (M)   61.1032   53.0418   2.341
Chevron (CVX)   64.3842   53.3270   2.178
Visa (V)   73.5130   55.2876   0.226
###AMD (AMD)   59.4151   49.3346   -1.537
Verizon (VZ)   59.5578   50.5703   -1.617
Google (GOOGL)   97.8768   50.9040   -2.981
J.P. Morgan (JPM)   63.4570   50.9506   -0.497
Exxon Mobil (XOM)   62.4822   51.5209   1.806
Bank of America (BAC)   61.3172   51.7110   -0.502
Macys (M)   61.1032   52.4715   2.667
Chevron (CVX)   64.3604   52.9468   2.239
Paypal (PYPL)   76.3314   54.4379   1.303
Visa (V)   73.5595   56.9573   0.367
###Google (GOOGL)   51.2704   47.4270   -4.147
Verizon (VZ)   52.2111   49.1445   -0.137
AMD (AMD)   53.2573   49.2395   -1.454
Chevron (CVX)   53.4475   49.5247   -1.602
Bank of America (BAC)   50.8321   50.0951   -0.073
Macys (M)   51.5692   50.9506   2.298
Exxon Mobil (XOM)   53.3761   51.5209   1.438
J.P. Morgan (JPM)   52.3062   53.0418   0.344
Paypal (PYPL)   51.4793   55.0296   0.000
Visa (V)   55.0186   57.8850   -0.148
###Macys (M)   51.3077   48.8593   0.616
AMD (AMD)   51.8545   50.1901   -1.435
Chevron (CVX)   53.2335   50.2852   -0.105
Verizon (VZ)   50.6419   51.0456   1.476
J.P. Morgan (JPM)   52.8293   51.5209   -0.563
Bank of America (BAC)   52.7342   51.9011   -0.842
Google (GOOGL)   52.2450   53.5466   0.238
Exxon Mobil (XOM)   53.4237   54.4677   1.559
Paypal (PYPL)   51.4793   55.0296   0.000
Visa (V)   52.8346   56.0297   -0.262"""
s=s.split("\n")
ss=[]
for i in s:
    ss.append(i.split("  "))
print(ss)

for i in ss:
    print(i[-1],i[-2],i[-3])
with open("testoutput.csv",'w+', newline='') as f:
    for i in ss:
        print(i[-4],i[-3],i[-2],i[-1])
        f.write(i[-4]+','+i[-3]+','+i[-2]+','+i[-1]+"\n")
