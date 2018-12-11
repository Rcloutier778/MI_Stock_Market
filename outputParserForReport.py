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
###Paypal (PYPL)   47.3373   42.6036   -1.507
Macys (M)   52.1636   48.9544   -0.156
AMD (AMD)   51.5692   50.0000   -1.203
Chevron (CVX)   52.9957   50.0951   -1.432
Bank of America (BAC)   52.0922   50.2852   -1.639
Google (GOOGL)   52.1754   51.0431   -1.127
Verizon (VZ)   51.5692   51.5209   0.678
Exxon Mobil (XOM)   53.0433   52.7567   1.543
J.P. Morgan (JPM)   51.6167   52.8517   -0.018
Visa (V)   54.7398   56.9573   -0.214
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
###Verizon (VZ)   51.8783   48.7643   -1.415
AMD (AMD)   52.4964   49.9049   -1.439
Google (GOOGL)   51.9318   50.6259   -2.215
Chevron (CVX)   52.8293   51.0456   0.424
Macys (M)   49.7147   51.2357   3.191
Bank of America (BAC)   52.4489   51.4259   -1.682
Exxon Mobil (XOM)   53.8516   52.2814   1.688
J.P. Morgan (JPM)   52.7580   52.2814   1.017
Paypal (PYPL)   51.6272   55.0296   0.000
Visa (V)   54.4610   60.6679   0.313"""
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
