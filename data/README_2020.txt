ETRI_Lifelog_Dataset_2020

Human Enhancement and Assistive Technology Research Section
AI Research Lab, Electronics and Telecommunications Research Institute, South Korea

================================================================================

To understand the multilateral characteristics of human behavioral and physiological markers related to physical, emotional, 
and contextual states, we performed long-term lifelog data collection experiments in a real-world environment. 
The processed dataset includes 570 days of experimental sessions, about 7,350 hours of data from 22 subjects.
It contains physiological data such as PPG, EDA, and skin temperature from a wrist-worn sensor (Empatica E4), in addition to
the multivariate behavioral data such as IMU (mobile phone and E4) and GPS data. 
The dataset consists of 440,830 processed labels (10,732 unique labels) that comprehend a broad range of everyday activities
(including mode of transportation) and contextual information such as semantic places and social states.
User labels also contain 2D (arousal-valence) emotional states using seven-point Likert scales. 

The dataset includes sensory data from the following sensors:
================================================================================
- Triaxial acceleration force (in m/s^2) from the mobile phone accelerometer (30 Hz) and E4 accelerometer (32 Hz)
- Triaxial rate of rotation (in rad/s) and degrees of rotation (in Degrees) from the mobile phone gyroscope (30 Hz)
- Triaxial geomagnetic field strength (in μT) from the mobile phone magnetometer (30 Hz)
- Latitude and longitude, and horizontal accuracy* (in meters) from the mobile phone GPS (every 5 seconds)
- Blood volume pressure (in nano Watt) from E4 photoplethysmography (PPG) sensor (64 Hz)
- Electrodermal activity (skin conductance in μS) from E4 EDA sensor (4 Hz)
- Average** heart rate values (in bps) computed in 10 seconds-span based on the BVP analysis from E4 (1 Hz)
- Peripheral skin temperature (in Celsius degrees) from E4 infrared thermopile (4 Hz)

* The estimated horizontal accuracy is defined as the radius of 68% confidence according to the API.
Reference: https://developer.android.com/reference/android/location/Location#getAccuracy()
** HR values are not derived from a real-time reading but are created after the data collection session.
Reference: https://support.empatica.com/hc/en-us/articles/360029469772-E4-data-HR-csv-explanation

The dataset includes files in a structure shown below:
================================================================================
+----- USER_ID
 |        +----- timestamp (DAY 1)
 |         |        +----- e4Acc
 |         |         |        timestamp (e4_accelerometer_data).csv 
 |         |         |        ...
 |         |        +----- e4Bvp
 |         |         |        timestamp (e4_blood_volume_pressure_data).csv 
 |         |         |        ...
 |         |        +----- e4Eda
 |         |         |        timestamp (e4_electrodermal_activity_data).csv 
 |         |         |        ...
 |         |        +----- e4Hr
 |         |         |        timestamp (e4_heart_rate_data).csv 
 |         |         |        ...
 |         |        +----- e4Temp
 |         |         |        timestamp (e4_skin_temperature_data).csv 
 |         |         |        ...
 |         |        +----- mAcc
 |         |         |        timestamp (mobile_accelerometer_data).csv 
 |         |         |        ...
 |         |        +----- mGps
 |         |         |        timestamp (mobile_gps_data).csv 
 |         |         |        ...
 |         |        +----- mGyr
 |         |         |        timestamp (mobile_gyroscope_data).csv 
 |         |         |        ...
 |         |        +----- mMag
 |         |         |        timestamp (mobile_magnetometer_data).csv 
 |         |         |        ...
 |         |        timestamp_label.csv
 |        +----- timestamp (DAY 2)
 |         |        +----- ...
================================================================================

Directories (in timestamps) located under the USER_ID directory indicate when the user started the experiment each day.
Each day has directories named by the corresponding sensors, which includes data files generated every one minute.
Each data file records raw sensor values in the designated sampling interval with the timestamp. 
(Timestamp is represented in second.millisecond format.)

User label files are composed of 12 columns representing the physical, emotional, and contextual states as follows:
================================================================================
- ts			timestamp
- action (16)		sleep, personal_care, work, study, household, care_housemem (caregiving), recreation_media, entertainment, 
			outdoor_act (sports), hobby, recreation_etc (free time), shop, communitiy_interaction (regular activity), travel, meal, socialising
- actionOption (73)		Details of the selected action. See the description below.
- actionSub		meal_amount when action=meal or snack
			move_method when action=travel
- actionSubOption		1 (light), 2 (moderate), 3 (heavy) when actionSub=meal_amount
			1 (walk), 2 (driving), 3 (taxi, passenger), 4 (personal mobility), 5 (bus), 6 (train, subway), 7 (others) when actionSub=move_method
- condition		ALONE, WITH_ONE, WITH_MANY
- conditionSub1Option	1 (with families), 2 (with friends), 3 (with colleagues), 4 (acquaintances), 5 (others)
- conditionSub2Option	1 (passive in conversation), 2 (moderate participation in conversation), 3 (active in conversation)
- place			home, workplace, restaurant, outdoor, other_indoor
- emotionPositive (Valence)	(negative) 1-2-3-4-5-6-7 (positive)
- emotionTension (Arousal)	(relaxed) 1-2-3-4-5-6-7 (aroused)
- activity***		0 (IN_VEHICLE), 1 (ON_BICYCLE), 2 (ON_FOOT), 3 (STILL), 4 (UNKNOWN), 5 (TILTING), 7 (WALKING), 8 (RUNNING)

*** Values in the activity column represent the detected activity of the mobile device using Google's Awareness API.
Reference: https://developers.google.com/android/reference/com/google/android/gms/location/DetectedActivity?hl=en

Descriptions for the actionOption field is as follows:
================================================================================
111	Sleep
112	Sleepless
121	Meal
122	Snack
131	Medical services, treatments, sick rest
132	Personal hygiene (bath)
133	Appearance management (makeup, change of clothes)
134	Beauty-related services
211	Main job
212	Side job
213	Rest during work
22	Job search
311	School class / seminar (listening)
312	Break between classes
313	School homework, self-study (individual)
314	Team project (in groups)
321	Private tutoring (offline)
322	Online courses
41	Preparing food and washing dishes
42	Laundry and ironing
43	Housing management and cleaning
44	Vehicle management
45	Pet and plant caring
46	Purchasing goods and services (grocery/take-out)
51	Caring for children under 10 who live together
52	Caring for elementary, middle, and high school students over 10 who live together
53	Caring for a spouse
54	Caring for parents and grandparents who live together
55	Caring for other family members who live together
56	Caring for parents and grandparents who do not live together
57	Caring for other family members who do not live together
81	Personal care-related travel
82	Commuting and work-related travel
83	Education-related travel
84	Travel related to housing management
85	Travel related to caring for family and household members
86	Travel related to participation and volunteering
87	Socializing and leisure-related travel
61	Religious activities
62	Political activity
63	Ceremonial activities
64	Volunteer
711	Offline communication
712	Video or voice call
713	Text or email (Online)
721	Reading books, newspapers, and magazines
722	Watching TV or video
723	Listening to audio
724	Internet search or blogging
725	Gaming (mobile, computer, video)
741	Watching a sporting event
742	Watching movie
743	Concerts and plays
744	Art galleries and museums
744	Amusement Park, zoo
745	Festival, carnival
746	Driving, sightseeing, excursion
751	Walking
752	Running, jogging
753	Climbing, hiking
754	Biking
755	Ball games (soccer, basketball, baseball, tennis, etc)
756	Personal exercises (yoga, pilates, etc.)
756	Camping, fishing
761	Group games (board games, card games, puzzles, etc.)
762	Personal hobbies (woodworking, gardening, etc.)
763	Group performances (orchestra, choir, troupe, etc.)
764	Liberal arts and learning (languages, musical instruments, etc.)
791	Nightlife
792	Smoking
793	Do nothing and rest
91	Online shopping
92	Offline shopping
================================================================================

If you use this dataset in the publication, please cite the following publication:
Seungeun Chung, Chi Yoon Jeong, Jeong Mook Lim, Jiyoun Lim, Kyoung Ju Noh, Gague Kim, Hyuntae Jeong, 
"Real-world Multimodal Lifelog Dataset for Human Behavior Study", ETRI Journal 2021 (to appear).

================================================================================

This work was supported by Electronics and Telecommunications Research Institute (ETRI) grant funded by the Korean government. 
[21ZS1100, Core Technology Research for Self-Improving Artificial Intelligence System]
The experiment was performed with Institutional Review Board (IRB) approval from the Korea National Institute for Bioethics Policy (KoNIBP).

================================================================================

Contact: s-c-h-u-n-g@e-t-r-i.r-e.k-r (Remove hyphens)