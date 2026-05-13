* 清洗部分

rename 总分 score
rename 企业年限 age
rename 注册资金万元 money
rename 企业人数 emp_num
rename 企业所有权性质 type
rename 企业所属行业类别 industry
rename 合作对象 parterner
rename 企业合作学校数量 pa_num


* 分析部分

drop if money == .
swilk score age money emp_num pa_num // 正态性检验（都没通过，所以用非参数检验和简单的均值比较即可）

// 分组的非参数检验

* 行业
kwallis score ,by ( industry )
kwallis y1 ,by ( industry )
kwallis y2 ,by ( industry )
kwallis y3 ,by ( industry )

tabstat score, by(industry) stat(mean)
tabstat y1, by(industry) stat(mean)
tabstat y2, by(industry) stat(mean)
tabstat y3, by(industry) stat(mean)

* 企业类型
kwallis score ,by ( type )
kwallis y1 ,by ( type )
kwallis y2 ,by ( type )
kwallis y3 ,by ( type )

tabstat score, by(type) stat(mean)
tabstat y1, by(type) stat(mean)
tabstat y2, by(type) stat(mean)
tabstat y3, by(type) stat(mean)

* 企业人数
gen enter_emp = 1 if emp_num <= 50
replace enter_emp = 2 if emp_num > 50 & emp_num <= 300
replace enter_emp = 3 if emp_num > 300

kwallis score ,by ( enter_emp )
kwallis y1 ,by ( enter_emp )
kwallis y2 ,by ( enter_emp )
kwallis y3 ,by ( enter_emp )

tabstat score, by(enter_emp) stat(mean)
tabstat y1, by(enter_emp) stat(mean)
tabstat y2, by(enter_emp) stat(mean)
tabstat y3, by(enter_emp) stat(mean)

* 企业合作学校类型
kwallis score ,by ( parterner )
kwallis y1 ,by ( parterner )
kwallis y2 ,by ( parterner )
kwallis y3 ,by ( parterner )

tabstat score, by(parterner) stat(mean)
tabstat y1, by(parterner) stat(mean)
tabstat y2, by(parterner) stat(mean)
tabstat y3, by(parterner) stat(mean)

* 合作学校数量
gen par_num_count = 1 if pa_num <= 2
replace par_num_count = 2 if pa_num > 2 & pa_num <= 5
replace par_num_count = 3 if pa_num > 5

kwallis score ,by ( par_num_count)
kwallis y1 ,by ( par_num_count )
kwallis y2 ,by ( par_num_count )
kwallis y3 ,by ( par_num_count )

tabstat score, by(par_num_count) stat(mean)
tabstat y1, by(par_num_count) stat(mean)
tabstat y2, by(par_num_count) stat(mean)
tabstat y3, by(par_num_count) stat(mean)

// 线性回归分析

reg score age money emp_num i.type i.industry i.parterner pa_num
estimates store m1

reg score age money emp_num i.type i.industry i.parterner pa_num,r
estimates store m2

winsor2 age money emp_num pa_num,cut(1,99) replace
reg score age money emp_num i.type i.industry i.parterner pa_num,r
estimates store m3

esttab m1 m2 m3  using C:,replace b(%12,3f) se(%12,3f) nogap compress /// 
 /// 
 s(N r2) star(* 0.1 ** 0.05 *** 0.01) // 导出回归结果到指定文件夹

egen std_moeny = std(money)
egen std_age = std(age)
egen std_emp_num = std(emp_num)
egen std_pa_num = std(pa_num)
reg score std_age std_money std_emp_num i.type i.industry i.parterner std_pa_num,r