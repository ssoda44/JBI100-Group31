1. Modifications

a. Baseline calculation is claimed by mean, but actually by median, so it is modified to mean.
b. Calculations of the previous version counts all the null data points, namely the 3N week staff presence, making the mean lower than ought to be. Now the now null data points are properly handled.

2. Diagnosis view

a. using the total demand and bed supply as the mean reasoning tool for analysis of refutation of patients or shortage rate. (because the staff number has no effect on it. The patients accepted is actually min(request, bed capacity)), if the demand delta is high, the cause may be the sudden increase of demand; if the bed supply delta is low, the cause maybe the sudden decrease of bed supply.
b. for the resources mismatch(bed management mismatch), a breakdown of the demand and the wastage of beds for all departments are displayed. using both bar charts and line chart. It can clearly show the wastage caused by a particular department.
c. A bed capacity breakdown is also provided for possible need of extra information.

3. interaction logic

After selecting a radio button in the Incidents view, the corresponding week number will be set for the view of Diagnosis for a smoother analysis process by the user. And a dashed line will highlight the corresponding week accross all the figures in the Diagnosis view.

...
