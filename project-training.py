import pyodbc
import csv
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from combinations import Solution


DBconnection = pyodbc.connect(driver='{SQL Server}',
                            server='localhost',
                            database='cap_project',
                            uid='sa',
                            pwd='wenda')

cursor = DBconnection.cursor()
print(cursor)

# This query fetch for Inputs
cursor.execute("""
SELECT a.[Procedure_Name], a.[Appt_Durantion], a.[Provider_Name], YEAR(a.[Appointment_Date]) _YEAR , MONTH(a.[Appointment_Date]) _MONTH, DAY(a.[Appointment_Date]) _DAY, DATEPART(HOUR, a.[Appointment_Date]) _HOUR, DATEPART(MINUTE, a.[Appointment_Date]) _MINUTE, DATEPART(WEEKDAY, a.[Appointment_Date]) _WEEKDAY, 
	CASE
		WHEN 
	(SELECT TOP 1 COUNT(*) FROM [cap_project].[dbo].[event] b 
	WHERE b.[CheckIn_Time] is not null AND b.[CheckIn_Time] < a.[CheckIn_Time] AND (b.[NoShow_Flag] is NULL AND b.[Canceled_Flag] is NULL) 
	GROUP BY b.[Patient_ID]
	HAVING b.[Patient_ID] = a.[Patient_ID]) IS NULL THEN 0
	ELSE (SELECT TOP 1 COUNT(*) FROM [cap_project].[dbo].[event] b 
	WHERE b.[CheckIn_Time] is not null AND b.[CheckIn_Time] < a.[CheckIn_Time] AND (b.[NoShow_Flag] is NULL AND b.[Canceled_Flag] is NULL) 
	GROUP BY b.[Patient_ID]
	HAVING b.[Patient_ID] = a.[Patient_ID])
	END as complete,
	CASE
		WHEN (SELECT TOP 1 COUNT(*) FROM [cap_project].[dbo].[event] c 
	WHERE c.[Appointment_Date] < a.[Appointment_Date] AND (c.[Canceled_Flag] is NOT NULL) 
	GROUP BY c.[Patient_ID]
	HAVING c.[Patient_ID] = a.[Patient_ID]) IS NULL 
		THEN 0
		ELSE (SELECT TOP 1 COUNT(*) FROM [cap_project].[dbo].[event] c 
	WHERE c.[Appointment_Date] < a.[Appointment_Date] AND (c.[Canceled_Flag] is NOT NULL) 
	GROUP BY c.[Patient_ID]
	HAVING c.[Patient_ID] = a.[Patient_ID])
	END as cancel,
	CASE
		WHEN (SELECT TOP 1 COUNT(*) FROM [cap_project].[dbo].[event] d 
	WHERE d.[Appointment_Date] < a.[Appointment_Date] AND (d.[NoShow_Flag] is NOT NULL) 
	GROUP BY d.[Patient_ID]
	HAVING d.[Patient_ID] = a.[Patient_ID]) IS NULL THEN 0
		ELSE (SELECT TOP 1 COUNT(*) FROM [cap_project].[dbo].[event] d 
	WHERE d.[Appointment_Date] < a.[Appointment_Date] AND (d.[NoShow_Flag] is NOT NULL) 
	GROUP BY d.[Patient_ID]
	HAVING d.[Patient_ID] = a.[Patient_ID])
	END AS noshow,
	CASE 
		WHEN a.[NoShow_Flag] = 'Y' THEN 'NOSHOW'
		WHEN a.[Canceled_Flag] = 'Y' THEN 'CANCELLED'
		WHEN a.[CheckIn_Time] IS NOT NULL THEN 'COMP'
	END as ans
  FROM [cap_project].[dbo].[event] a
  WHERE a.[NoShow_Flag] = 'Y' OR a.[Canceled_Flag] = 'Y' OR a.[CheckIn_Time] IS NOT NULL
""")

# This query fetch for outputs

rows = cursor.fetchall()  # The SQL query result
ROWS = np.array(rows)  # transform query to np array
print("Rows")
print(ROWS[0])

# ------------------------Prepare Data-----------------------------
procedure_name = ROWS[:, 0]  # vectorize
appt_duration = ROWS[:, 1]  # discrete
provider_name = ROWS[:, 2]  # vectorize

# vectorize by year, month, day, hours, and minutes
appointment_year = ROWS[:, 3].astype(np.string_)
appointment_month = ROWS[:, 4].astype(np.string_)
appointment_day = ROWS[:, 5].astype(np.string_)
appointment_hour = ROWS[:, 6].astype(np.string_)
appointment_minute = ROWS[:, 7].astype(np.string_)
appointment_weekday = ROWS[:, 8].astype(np.string_)

complete = ROWS[:, 9]  # discrete
cancel = ROWS[:, 10]  # discrete
noshow = ROWS[:, 11]  # discrete

answer = ROWS[:, 12]  # The answer
# Provide list of datetime categories
yearlist = np.array(range(1970, 2050)).astype(np.string_)
monthlist = np.array(range(1, 13)).astype(np.string_)
daylist = np.array(range(1, 32)).astype(np.string_)
hourlist = np.array(range(8, 17)).astype(np.string_)
# Minutes with step size of 5, like 15, 20, 30, 45, etc...
minutelist = np.array(range(1, 61, 5)).astype(np.string_)
weeklist = np.array(range(1, 8)).astype(np.string_)
# ------------------------End Prepare Data-----------------------------

# ------------------------Vectorize-------------------------------
vectorizer = LabelBinarizer()  # Vectorize by going through the SQL result

procedure_name_vec = vectorizer.fit_transform(procedure_name)
print("Procedure Name")
print(procedure_name_vec[0])
provider_name_vec = vectorizer.fit_transform(provider_name)
print("Provider Name")
print(provider_name_vec[0])
answer_vec = vectorizer.fit_transform(answer)  # This is the output
print("Answer shape")
print(answer_vec.shape)

appt_duration_vec = appt_duration[:, None]
complete_vec = complete[:, None]
cancel_vec = cancel[:, None]
noshow_vec = noshow[:, None]

lb_year = LabelBinarizer().fit(yearlist)
lb_month = LabelBinarizer().fit(monthlist)
lb_day = LabelBinarizer().fit(daylist)
lb_hour = LabelBinarizer().fit(hourlist)
lb_minute = LabelBinarizer().fit(minutelist)
lb_week = LabelBinarizer().fit(weeklist)
# In format of [0, 0, 0, 1, 0...]
vect_year = lb_year.transform(list(appointment_year))
vect_month = lb_month.transform(list(appointment_month))
vect_day = lb_day.transform(list(appointment_day))
vect_hour = lb_hour.transform(list(appointment_hour))
vect_minute = lb_minute.transform(list(appointment_minute))
vect_weekday = lb_week.transform(list(appointment_weekday))
# --------------------------End Vectorize--------------------------------

# -----------------------Feature (as Input)-------------------
dict_features = {0: procedure_name_vec,
                 1: provider_name_vec,
                 2: appt_duration_vec,
                 3: vect_year,
                 4: vect_month,
                 5: vect_day,
                 6: vect_hour,
                 7: vect_minute,
                 8: vect_weekday,
                 9: complete_vec,
                 10: cancel_vec,
                 11: noshow_vec
                 }
# -----------------------End Feature (as Input)-------------------

dict_features_names = {0: "Procedure Name",
                 1: "Provider Name",
                 2: "Appointment Duration",
                 3: "year",
                 4: "Month",
                 5: "Day",
                 6: "Hour",
                 7: "Minute",
                 8: "Weekday",
                 9: "Completed Counts",
                 10: "Canceled Counts",
                 11: "No show Counts"}

record_acc_file_name = "accuracy.csv"
if os.path.exists(record_acc_file_name):
    append_write = "w"
else:
    append_write = "w"
record_acc_file = open(record_acc_file_name, append_write, newline='')
record_acc_file_writer = csv.writer(record_acc_file)
record_acc_file_writer.writerow(['Combinations_id', 'Combinations_name', 'Avg_Accuracy'])

solution = Solution()
highest_accuracy = 0
for i in range(1, len(dict_features)):  # Loop, i controls the element numbers of features
    for result in solution.combine(12, i + 1):
        print("running with %i of combinations" % (i + 1))
        result[:] = [x - 1 for x in result]  # Subtract all values in result by 1, to match the id (0)
        temp_tuple_features = ()
        print("result names: ", end="")
        for id in result:
            print(dict_features_names[id])
            # TODO: fix errors of the wrong tuple shape, try to replace the tuple_features with dictionary type in order to select each individual items
            temp_tuple_features += (dict_features[id], )

        # temp_tuple_features = map(tuple_features.__getitem__, result)

        if i > 0:
            feature_data = np.concatenate(temp_tuple_features, axis=1).astype(np.float)
            print("i != 0")
        elif i == 0:
            feature_data = temp_tuple_features[0]

        print("Feature Data Shape")
        print(feature_data.shape)

        train_accuracy = 0
        for i in range(30):
            train_x, test_x, train_y, test_y = train_test_split(feature_data,
                                                                answer_vec,
                                                                test_size=0.3)
            clf = MLPClassifier(solver='lbfgs',
                                alpha=1e-5,
                                hidden_layer_sizes=(30, 20, 10, 5),
                                random_state=1)
            clf.fit(train_x, train_y)
            train_accuracy += clf.score(test_x, test_y, sample_weight=None)
        avg_acc = train_accuracy / 30
        print("Mean Accuracy: %f " % avg_acc)
        if avg_acc > highest_accuracy:
            highest_accuracy = avg_acc
        print("Highest Accuracy so far: %f" % highest_accuracy)
        print("When result is ", end = "")
        print(result)

        combinations_id = ', '.join(str(result_id) for result_id in result)
        combinations_name = ', '.join(dict_features_names[result_id] for result_id in result)
        record_acc_file_writer.writerow([combinations_id, combinations_name, avg_acc])
record_acc_file.close()

# feature_data = np.concatenate((procedure_name_vec,
#                                 provider_name_vec,
#                                 appt_duration_vec,
#                                 vect_year,
#                                 vect_month,
#                                 vect_day,
#                                 vect_hour,
#                                 vect_minute,
#                                 vect_weekday,
#                                 complete_vec,
#                                 cancel_vec,
#                                 noshow_vec
#                                 ), axis=1).astype(np.float)
# print(feature_data[0])