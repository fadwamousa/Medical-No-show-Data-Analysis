#!/usr/bin/env python
# coding: utf-8

# ## introduction

# Medical Appointment No Shows
# Why do 30% of patients miss their scheduled appointments?

# ### Questions to Ask:

# - What is the percentage of patients who show up on their appointements vs. who don't?

# - Do certain gender has more commitment to medical schedules than the other one?

# - Is the duration between regestiration and appointment affect the ability to show up ?

# - Do patients who recieves SMS to remind them of the appointement more likely to show up?

# - What is percentage of patients who diagnosed with Diabetes, Hipertension, Alcoholism, and Handcap ?

# - Is alcohol drinking may be a cause of missing out the appointements?

# In[1]:


# import importnat library for analyzie
import pandas as pd
import numpy  as  np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Wrangling

# #### General Properties
# - Assesing

# In[2]:


df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# In[3]:


df.shape[0] #number of smaples


# In[4]:


df.info()


# - we have a dataset with 110527 entries and 14 varaibles 
# - we don't have any null in dataset
# - PatientId should convert form Float into Int datatype
# - AppointmentDay & ScheduledDay

# #### Missing 

# In[5]:


df.isnull().sum()


# #### Duplicates 

# In[8]:


sum(df.duplicated())


# In[13]:


df.sample(10)


# #### there is age = 0 !! (This is outliers) must be removed

# In[14]:


#get statistical data about each column
df.describe()


# - Age has a one or more negative value which makes no sense, so it need to be removed.

# In[19]:


df.hist(figsize=(15,15));
plt.tight_layout()


# - Age include more young patients
# - A small percentage of patients suffer from diabetes, high blood pressure , alcohol addiction and Handcap
# - SMS have been sent to more than 30% of cases

# ### Data Cleaning

# - Edit the "No-show" Column to be in Positive form instead of Negativity

# - 0 will mean that patient didn't come to his appointement.
# - 1 will mean that patient came to his appointment.

# - Rename the column No_show into Show_up

# In[29]:


df['No-show'].replace({'Yes':0 , 'No':1} , inplace = True)


# In[31]:


df.rename(columns = {'No-show' : 'Show_up'} , inplace = True)


# In[34]:


df.Show_up.dtypes


# 
# - Edit the "ScheduledDay", and "AppointmentDay" Columns' Datatype to be Datetime

# In[37]:


df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay'])


# In[38]:


df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])


# - Edit the "PatientId", "AppointmentID" Columns' Datatype to be String

# In[51]:


df['PatientId'].dtypes


# In[50]:


df['AppointmentID'].dtypes


# In[126]:


df['AppointmentID'] = df['AppointmentID'].apply('str')
#df['AppointmentID'].astype('object')


# In[127]:


df['PatientId'] = df['PatientId'].apply('str')
#df['PatientId'].astype('object')


# - Remove Row(s) with Negative Age Value(s):

# In[83]:


sum(df['Age'] < 0)


# In[84]:


sum(df["Age"]<0) 


# - Let's convert them into nan then drop them

# In[71]:


df['Age'] = df['Age'].replace(0,np.nan)


# ##### Test

# In[87]:


sum(df["Age"] == 0) 


# In[75]:


df['Age'].dropna(inplace = True)


# In[77]:


df.isnull().sum()


# In[80]:


df.dropna(subset = ['Age'] , inplace = True)


# In[89]:


df[df['Age'] < 0]


# In[ ]:


df.drop([99832],inplace = True)


# ## Exploratory Data Analysis

# - (What is the percentage of patients who show up on their appointements vs. who don't?)

# In[106]:


df.groupby('Show_up').count()["PatientId"]


# In[108]:


df.groupby('Show_up').count()["PatientId"].plot(kind = 'bar');


# In[104]:


show_yes = df[df['Show_up'] == 1].count()['PatientId']
show_no  = df[df['Show_up'] == 0].count()['PatientId']


# In[105]:


label_Names = ["Showed up", "Didn't Show up"]
data = [show_yes, show_no]


explode = (0, 0.15) #only explode the didn't show up slice.
plt.axis('equal'); #to keep aspect ratio equal to appear as a fine circle.
plt.pie(data,radius=1.5,shadow=True ,labels = label_Names,explode=explode, startangle=180,autopct='%0.2f%%',textprops = {"fontsize":15})
plt.title("Percentage of patients who showed up and who didn't",y=1.2);
#autopct to show percentage, 0.2 for two decimal place


# -  (Do certain gender has more commitment to medical schedules than the other one?)

# In[116]:


#df.groupby('Gender').count()
Female = df[df['Gender'] == 'F'].count()['PatientId']
Male   = df[df['Gender'] == 'M'].count()['PatientId']


# In[121]:


print("The Female number is : " + str(Female) + "And the Male number is: "+str(Male))


# In[122]:


label_Names = ["Female", "Male"]
data = [Female, Male]


explode = (0, 0.15) #only explode the didn't show up slice.
plt.axis('equal'); #to keep aspect ratio equal to appear as a fine circle.
plt.pie(data,radius=1.5,shadow=True ,labels = label_Names,explode=explode, startangle=180,autopct='%0.2f%%',textprops = {"fontsize":15})
plt.title("Percentage of patients who showed up and who didn't",y=1.2);
#autopct to show percentage, 0.2 for two decimal place


# -  (Is the duration between registeration and appointment affect the ability to show up ?)

# In[123]:


df['Duration'] = (df['AppointmentDay'].dt.date) - (df['ScheduledDay'].dt.date)


# In[134]:


df["Duration"] = df["Duration"].dt.days


# In[135]:


df.groupby('Show_up').mean()['Duration']


# - Patients Who didn't show up have an average of 15 days between registeration day and their appointments.
# 
# - Patients Who show up have an average of 8 days between registeration day and their appointments.
# 
# - As Duration increases, the ability of patients to show up on their appointments decreases.
# 
# 

# - Do patients who recieves SMS to remind them of the appointement more likely to show up?

# In[146]:


df_SMS    = df[df["SMS_received"] == 1]["Show_up"].mean()


# In[147]:


df_No_SMS = df[df["SMS_received"] == 0]["Show_up"].mean()


# In[150]:


label_Names = ["SMS", "NO_SMS"]
data = [df_SMS, df_No_SMS]


explode = (0, 0.15) #only explode the didn't show up slice.
plt.axis('equal'); #to keep aspect ratio equal to appear as a fine circle.
plt.pie(data,radius=1.5,shadow=True ,labels = label_Names,explode=explode, startangle=180,autopct='%0.2f%%',textprops = {"fontsize":15});


# - What is percentage of patients who diagnosed with Diabetes, Hipertension, Alcoholism, and Handcap?

# In[151]:


all_count = 110526 #number of data records


# In[156]:


diabets_count = df[df["Diabetes"]==1]["PatientId"].count()


# In[161]:


#percentage of people diagnosed with Diabetes
diabetes_percent = round(diabets_count *100 / all_count , 2)


# In[162]:


diabetes_percent


# In[163]:


print("Percentage of patients who diagnosed with Diabetes is {}%.".format(diabetes_percent))


# In[164]:


#percentage of people diagnosed with Diabetes
diabets_count = df[df["Hipertension"]==1]["PatientId"].count()
diabetes_percent = round(diabets_count*100/all_count,2)
print("Percentage of patients who diagnosed with Hipertension is {}%.".format(diabetes_percent))


# In[165]:


#percentage of people diagnosed with Diabetes
diabets_count = df[df["Alcoholism"]==1]["PatientId"].count()
diabetes_percent = round(diabets_count*100/all_count,2)
print("Percentage of patients who diagnosed with Alcoholism is {}%.".format(diabetes_percent))


# In[166]:


#percentage of people diagnosed with Diabetes
diabets_count = df[df["Handcap"]==1]["PatientId"].count()
diabetes_percent = round(diabets_count*100/all_count,2)
print("Percentage of patients who diagnosed with Handcap is {}%.".format(diabetes_percent))


# - Is alcohol drinking may be a cause of missing out the appointements?

# In[174]:


#count of patients with alchoholism who show up
alcohol_show = df.loc[(df["Alcoholism"]==1) & (df["Show_up"]==1)]["PatientId"].count() 


# In[177]:


#count of patients with alchoholism who don't show up
alcohol_No_show = df.loc[(df["Alcoholism"]==1) & (df["Show_up"]==0)]["PatientId"].count() 


# In[178]:


alcohol_show_percent = round(alcohol_show*100 / all_count,2) 
alcohol_No_show_percent = round(alcohol_No_show*100 / all_count,2)


# In[179]:


print("Percentage of show ups when patients have alchoholism is {}%, while not show ups is {}%.".format(alcohol_show_percent,alcohol_No_show_percent))


# In[ ]:




