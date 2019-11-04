SELECT 
	AVG(val1),
	AVG(val2),
	AVG(val3),
	AVG(val4),
	AVG(val5),
	AVG(val6),
	AVG(val7),
	AVG(val8),
	AVG(val9),
	AVG(val10),
	AVG(val11),
	AVG(val12),
	AVG(val13),
	AVG(val14),
	AVG(val15),
	AVG(val16),
	AVG(val17),
	AVG(val18),
	AVG(val19),
	AVG(val20),
	AVG(val21),
	AVG(val22),
	AVG(val23),
	AVG(val24),
	AVG(val25),
	AVG(val26),
	AVG(val27),
	AVG(val28),
	AVG(val29),
	AVG(val30),
	AVG(val31),
	AVG(val32),
	AVG(val33),
	AVG(val34),
	AVG(val35),
	AVG(val36),
	AVG(val37),
	AVG(val38),
	AVG(val39),
	AVG(val40),
	AVG(val41),
	AVG(val42),
	AVG(val43),
	AVG(val44),
	AVG(val45),
	AVG(val46),
	AVG(val47),
	AVG(val48),
	AVG(val49),
	AVG(val50),
	AVG(val51),
	AVG(val52),
	AVG(val53),
	AVG(val54),
	AVG(val55),
	AVG(val56),
	AVG(val57),
	AVG(val58),
	AVG(val59),
	AVG(val60)
FROM 
(
	SELECT 
		cp.date1  + ccy1.date1  * ccy1_param.date1   - ccy2.date1  * ccy2_param.date1 AS val1,
		cp.date2  + ccy1.date2  * ccy1_param.date2   - ccy2.date2  * ccy2_param.date2 AS val2,
		cp.date3  + ccy1.date3  * ccy1_param.date3   - ccy2.date3  * ccy2_param.date3 AS val3,
		cp.date4  + ccy1.date4  * ccy1_param.date4   - ccy2.date4  * ccy2_param.date4 AS val4,
		cp.date5  + ccy1.date5  * ccy1_param.date5   - ccy2.date5  * ccy2_param.date5 AS val5,
		cp.date6  + ccy1.date6  * ccy1_param.date6   - ccy2.date6  * ccy2_param.date6 AS val6,
		cp.date7  + ccy1.date7  * ccy1_param.date7   - ccy2.date7  * ccy2_param.date7 AS val7,
		cp.date8  + ccy1.date8  * ccy1_param.date8   - ccy2.date8  * ccy2_param.date8 AS val8,
		cp.date9  + ccy1.date9  * ccy1_param.date9   - ccy2.date9  * ccy2_param.date9 AS val9,
		cp.date10 + ccy1.date10 * ccy1_param.date10 - ccy2.date10 * ccy2_param.date10 AS val10,
		cp.date11 + ccy1.date11 * ccy1_param.date11 - ccy2.date11 * ccy2_param.date11 AS val11,
		cp.date12 + ccy1.date12 * ccy1_param.date12 - ccy2.date12 * ccy2_param.date12 AS val12,
		cp.date13 + ccy1.date13 * ccy1_param.date13 - ccy2.date13 * ccy2_param.date13 AS val13,
		cp.date14 + ccy1.date14 * ccy1_param.date14 - ccy2.date14 * ccy2_param.date14 AS val14,
		cp.date15 + ccy1.date15 * ccy1_param.date15 - ccy2.date15 * ccy2_param.date15 AS val15,
		cp.date16 + ccy1.date16 * ccy1_param.date16 - ccy2.date16 * ccy2_param.date16 AS val16,
		cp.date17 + ccy1.date17 * ccy1_param.date17 - ccy2.date17 * ccy2_param.date17 AS val17,
		cp.date18 + ccy1.date18 * ccy1_param.date18 - ccy2.date18 * ccy2_param.date18 AS val18,
		cp.date19 + ccy1.date19 * ccy1_param.date19 - ccy2.date19 * ccy2_param.date19 AS val19,
		cp.date20 + ccy1.date20 * ccy1_param.date20 - ccy2.date20 * ccy2_param.date20 AS val20,
		cp.date21 + ccy1.date21 * ccy1_param.date21 - ccy2.date21 * ccy2_param.date21 AS val21,
		cp.date22 + ccy1.date22 * ccy1_param.date22 - ccy2.date22 * ccy2_param.date22 AS val22,
		cp.date23 + ccy1.date23 * ccy1_param.date23 - ccy2.date23 * ccy2_param.date23 AS val23,
		cp.date24 + ccy1.date24 * ccy1_param.date24 - ccy2.date24 * ccy2_param.date24 AS val24,
		cp.date25 + ccy1.date25 * ccy1_param.date25 - ccy2.date25 * ccy2_param.date25 AS val25,
		cp.date26 + ccy1.date26 * ccy1_param.date26 - ccy2.date26 * ccy2_param.date26 AS val26,
		cp.date27 + ccy1.date27 * ccy1_param.date27 - ccy2.date27 * ccy2_param.date27 AS val27,
		cp.date28 + ccy1.date28 * ccy1_param.date28 - ccy2.date28 * ccy2_param.date28 AS val28,
		cp.date29 + ccy1.date29 * ccy1_param.date29 - ccy2.date29 * ccy2_param.date29 AS val29,
		cp.date30 + ccy1.date30 * ccy1_param.date30 - ccy2.date30 * ccy2_param.date30 AS val30,
		cp.date31 + ccy1.date31 * ccy1_param.date31 - ccy2.date31 * ccy2_param.date31 AS val31,
		cp.date32 + ccy1.date32 * ccy1_param.date32 - ccy2.date32 * ccy2_param.date32 AS val32,
		cp.date33 + ccy1.date33 * ccy1_param.date33 - ccy2.date33 * ccy2_param.date33 AS val33,
		cp.date34 + ccy1.date34 * ccy1_param.date34 - ccy2.date34 * ccy2_param.date34 AS val34,
		cp.date35 + ccy1.date35 * ccy1_param.date35 - ccy2.date35 * ccy2_param.date35 AS val35,
		cp.date36 + ccy1.date36 * ccy1_param.date36 - ccy2.date36 * ccy2_param.date36 AS val36,
		cp.date37 + ccy1.date37 * ccy1_param.date37 - ccy2.date37 * ccy2_param.date37 AS val37,
		cp.date38 + ccy1.date38 * ccy1_param.date38 - ccy2.date38 * ccy2_param.date38 AS val38,
		cp.date39 + ccy1.date39 * ccy1_param.date39 - ccy2.date39 * ccy2_param.date39 AS val39,
		cp.date40 + ccy1.date40 * ccy1_param.date40 - ccy2.date40 * ccy2_param.date40 AS val40,
		cp.date41 + ccy1.date41 * ccy1_param.date41 - ccy2.date41 * ccy2_param.date41 AS val41,
		cp.date42 + ccy1.date42 * ccy1_param.date42 - ccy2.date42 * ccy2_param.date42 AS val42,
		cp.date43 + ccy1.date43 * ccy1_param.date43 - ccy2.date43 * ccy2_param.date43 AS val43,
		cp.date44 + ccy1.date44 * ccy1_param.date44 - ccy2.date44 * ccy2_param.date44 AS val44,
		cp.date45 + ccy1.date45 * ccy1_param.date45 - ccy2.date45 * ccy2_param.date45 AS val45,
		cp.date46 + ccy1.date46 * ccy1_param.date46 - ccy2.date46 * ccy2_param.date46 AS val46,
		cp.date47 + ccy1.date47 * ccy1_param.date47 - ccy2.date47 * ccy2_param.date47 AS val47,
		cp.date48 + ccy1.date48 * ccy1_param.date48 - ccy2.date48 * ccy2_param.date48 AS val48,
		cp.date49 + ccy1.date49 * ccy1_param.date49 - ccy2.date49 * ccy2_param.date49 AS val49,
		cp.date50 + ccy1.date50 * ccy1_param.date50 - ccy2.date50 * ccy2_param.date50 AS val50,
		cp.date51 + ccy1.date51 * ccy1_param.date51 - ccy2.date51 * ccy2_param.date51 AS val51,
		cp.date52 + ccy1.date52 * ccy1_param.date52 - ccy2.date52 * ccy2_param.date52 AS val52,
		cp.date53 + ccy1.date53 * ccy1_param.date53 - ccy2.date53 * ccy2_param.date53 AS val53,
		cp.date54 + ccy1.date54 * ccy1_param.date54 - ccy2.date54 * ccy2_param.date54 AS val54,
		cp.date55 + ccy1.date55 * ccy1_param.date55 - ccy2.date55 * ccy2_param.date55 AS val55,
		cp.date56 + ccy1.date56 * ccy1_param.date56 - ccy2.date56 * ccy2_param.date56 AS val56,
		cp.date57 + ccy1.date57 * ccy1_param.date57 - ccy2.date57 * ccy2_param.date57 AS val57,
		cp.date58 + ccy1.date58 * ccy1_param.date58 - ccy2.date58 * ccy2_param.date58 AS val58,
		cp.date59 + ccy1.date59 * ccy1_param.date59 - ccy2.date59 * ccy2_param.date59 AS val59,
		cp.date60 + ccy1.date60 * ccy1_param.date60 - ccy2.date60 * ccy2_param.date60 AS val60
FROM client_portfolios AS cp
	LEFT JOIN ccy1 ON cp.portfolio_id = ccy1.portfolio_id AND cp.sim_idx = ccy1.sim_idx
	LEFT JOIN ccy2 ON cp.portfolio_id = ccy2.portfolio_id AND cp.sim_idx = ccy2.sim_idx
	LEFT JOIN ccy1_param ON cp.portfolio_id = ccy1_param.portfolio_id AND cp.sim_idx = ccy1_param.sim_idx
	LEFT JOIN ccy2_param ON cp.portfolio_id = ccy2_param.portfolio_id AND cp.sim_idx = ccy2_param.sim_idx
);