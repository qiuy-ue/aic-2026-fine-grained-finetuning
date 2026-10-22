import os,struct,collections,sys
D=sys.argv[1] if len(sys.argv)>1 else r"D:\aic_data\dataset\test"
if not os.path.isdir(D): D="/workspace/D:/aic_data/dataset/test"
c=collections.Counter(); ls=collections.Counter(); n=0; bad=[]
for name in sorted(os.listdir(D)):
    with open(os.path.join(D,name),'rb') as f: buf=f.read(4096)
    i=2
    while i<len(buf)-1:
        if buf[i]!=0xFF: i+=1; continue
        m=buf[i+1]
        if m==0xFF: i+=1; continue
        if m in (0xD8,0x01) or 0xD0<=m<=0xD7: i+=2; continue
        L=struct.unpack(">H",buf[i+2:i+4])[0]
        if m in (0xC0,0xC1,0xC2,0xC3):
            h=struct.unpack(">H",buf[i+5:i+7])[0]; w=struct.unpack(">H",buf[i+7:i+9])[0]
            c[(w,h)]+=1; L2=max(w,h); ls[L2]+=1; n+=1
            if L2!=500: bad.append((name,w,h))
            break
        i=i+2+L
print("n=%d  长边==500: %d (%.4f%%)  长边!=500: %d (%.4f%%)"%(n, ls[500],100*ls[500]/n, n-ls[500],100*(n-ls[500])/n))
print("长边分布 top15:", ls.most_common(15))
print("短边 <500 的最小值们:", sorted(ls)[:8])
print("非500 的例子:", bad[:15])
sb=sorted(min(w,h) for (w,h),v in c.items() for _ in range(v))
print("短边 min/p1/p10/median/p90/max = %d/%d/%d/%d/%d/%d"%(sb[0],sb[len(sb)//100],sb[len(sb)//10],sb[len(sb)//2],sb[9*len(sb)//10],sb[-1]))
print("短边>=512 占比 %.4f%%"%(100*sum(1 for x in sb if x>=512)/len(sb)))
