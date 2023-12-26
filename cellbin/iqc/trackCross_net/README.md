# 关于Rotation  
角度方向（在OpenCV中u对应x，v对应y）：  
![avatar](https://img-blog.csdnimg.cn/20190207103137349.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d3d2x5ajEyMzMyMQ==,size_16,color_FFFFFF,t_70)  
取值范围：(-90, 90]  
截距：  
 x截距：(-90, -45) & (45, 90]  
 y截距：[-45, 45]  

唯一性保证（正切值曲线）：  
![avatar](https://i02piccdn.sogoucdn.com/1d2bc9d75313813d)

# 关于Scale  
定义：track线间距与标准芯片模板线间距之比。  
<font color=#1589F0 size=5> 更新（2021-08-16）：标准芯片模板线间距和track线间距之比。</font>  

# 关于索引匹配  
匹配对象：track线和模板间距  
说明：对于某个方向（X或Y），定有某相邻track线间距 = template[i] * Scale。  
定义：  
   &emsp;模板某方向第i个间距为，Ti  
   &emsp;FOV上track线及其右邻居track线：L(n)和L(n+1)  
   &emsp;track线距离：d = D(L(n), L(n+1))  
则track线L(n)在模板上的索引为：  
   &emsp;i in [0, 8] st. template[i] * scale = d.  