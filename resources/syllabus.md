## CSCI 4831-002: Machine Learning, Spring 2018

### Course Information 

**Class Meeting Time**: MWF 2-2:50pm in ECCR 200 

**Office Hours**: WF 12-1pm and Th 1:30-3pm in ECOT 731 or by appointment

**Prerequisites**: 
- C- or better in Algorithms 
- C- or better in CSCI 2820 (or equivalent)
- C- or better in CSCI 3022 (or equivalent)
- ability (or willingness to learn) to program in Python 3 

### Course Description 

Computers have made it possible, even easy, to collect vast amounts of data from a wide variety
of sources. It is not always clear, however, how to use those data and how to extract useful information from
data. This problem is faced in a tremendous range of scholarly, government, business, medical, and scientific
applications. The goal of this course is to review the principles of allowing machines to make sense of these data
in a mathematically rigorous way. By the end of this course, you'll be able to take a problem and analyze it to
determine which machine learning techniques are appropriate for solving the problem, how to prepare data to use
that solution, apply the solution, and to evaluate the results. For the most common machine learning techniques,
you'll also be able to implement solutions in Python.

### Textbook 

The main textbook for the course will be [Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) by James et. al. 

### Course Web Page 

[https://piazza.com/colorado/spring2018/csci4622](https://piazza.com/colorado/spring2017/csci4622)

This term we will be using Piazza for class discussion. The system is highly catered to getting you help fast and efficiently from classmates, the grader, and the instructor. **Rather than emailing me questions, I request that you to post your questions on Piazza**.  If your question is of a private nature, Piazza allows you to send me a private message. It is your responsibility to check the web page on a regular basis. Here you will find detailed information such as news, homework assignments and solutions, and instructor office hours. 

### Computing

A major component of this course is to learn modern, practical computing skills for machine learning.  Our two main tools will be Python 3 and Jupyter Notebooks.  It is estimated that around 50% of practicing data scientists do most of their analysis using Python (while the other roughly 50% use R, which we shall not speak of).  The Jupyter Notebook is a browser-based programming environment that allows you to seamlessly mix Python code (as well as many other languages), graphics, and exposition (in Markdown).  The Jupyter Notebook has become ubiquitous in the data science community for rapidly prototyping ideas and sharing them with colleagues and the rest of the world. 

It is strongly recommended that you install Python 3 and Jupyter on your local machine. By far, the easiest way to do this is by installing the [Anaconda](https://www.continuum.io/downloads) distribution of Python 3.6.  This distribution comes with many Python packages useful for data science and scientific computation in general. It also comes with Jupyter by default.

Frequently in class we will explore computational problems directly in Jupyter notebooks.  It is highly recommended that you bring a laptop with you to class, however it's perfectly acceptable that you team up with a classmate on a single computer. 

### Coursework 

**Homework**: Homework will be assigned roughly every two weeks and due by 5pm on Fridays on Moodle.  Assignments will be a mix of theoretical and computational problems.  The theoretical problems may include by-hand computations and simple proofs.  The computational problems will involve implementing and analyzing your own predictive models in Python or using canned routines from Scikit-Learn.  All assignments will be completed and submitted in Jupyter Notebooks.  You are allotted **THREE** late days that may be used for homework over the entire semester. Submitting an assignment between 1 second and 23 hrs, 59 minutes, 59 seconds late constitutes 1 late day.  After you have expended your allotted late days late homework will **NOT** be accepted or graded. **Your lowest homework score will be dropped**. You are expected to write up your solutions neatly, with full explanations and justifications. You may discuss problems with your classmates, **but all work must be your own**.  See the **Collaboration Policy** below for more details. 

**Reading Quizzes**: Most lectures will have an associated online "quiz" on Moodle due before the next class day.  These will consist of 
one or two short problems based on the reading and lecture material from the previous lecture.  You may use any reasonable resources 
available to you (lecture slides, class notes, reading material, your computer, the internet) but you should work the problems on your own.  These will not be terribly difficult, and are primarily designed to keep you honest with respect to attending class and keeping up with the reading.  **No extensions or make-up quizzes will be given** but I will drop 10% of your lowest quiz scores (e.g. if we end up
having 30 reading quizzes then I will drop your three lowest scores.)

**Exams**: There will be an evening midterm on March 7th and a final exam given during the university scheduled final examination time.  The final exam will be **cumulative** but will emphasize material covered after the midterm. Note that due to university regulations, **final exams can only be rescheduled due to official university excused absences**.  Please plan your holiday travel accordingly.  

**Exam Cut-Off**: You must earn a combined exam average of a 55% or greater to earn above a D+ in this course.  If the combined class average of the two exams is below a 75% then the 55% threshold will be adjusted down accordingly. 

**Practicum**: You will be given a practicum during the last two weeks of the semester.  This will be like a cumulative larger homework assignment where you show off all of the skills that you have learned in the course.  The practicum must be completed **completely independently**.  You may not discuss your solutions with anyone else in the class. 

### Grade Determination 

The overall grades will be based on a cumulative score computed from 

* The homework (30% of the grade)
* The reading quizzes (10% of the grade)
* The midterm exam (20% of the grade)
* The final exam (25% of the grade)
* The practicum (15% of the grade)


Unless adjustments are necessary, letter grades will be assigned using the standard grading scale: 

| Letter | Raw Average |
|--------|-------------|
|     A  |   93-100    |
|     A- |   90-92     |
|     B+ |   87-89     |
|     B  |   83-86     |
|     B- |   80-82     |
|     C+ |   77-79     |
|     C  |   73-76     |
|     C- |   70-72     |
|     D+ |   67-69     |
|     D  |   63-66     |
|     D- |   60-62     |
|     F  |   00-59     |

### Collaboration Policy 

The collaboration policy is simple:

* **Inspiration is free**: you may discuss homework assignments with anyone. You are especially encouraged to discuss solutions with your instructor and your classmates.

* **Plagiarism is forbidden**: the assignments **and code** that you turn in should be written entirely on your own. You should not need to consult sources beyond your textbook, class notes, posted lecture slides and notebooks, Python/Scikit-Learn/Numpy documentation, and online sources for _basic_ techniques. Copying/soliciting a solution to a problem from the internet or another classmate constitutes a violation of the course's collaboration policy and the honor code and will result in an **F** in the course and a trip to the honor council.

* **Do not search for a solution on-line**: You may not actively search for a solution to the problem from the internet. This includes posting to sources like StackExchange, Reddit, Chegg, etc. 

* **StackExchange Clarification**: Searching for basic techniques in Python/Pandas/Numpy is totally fine.  If you want to post and ask "How do I group by two columns, then do something, then group by a third column" that's fine.  What you **cannot** do is post "Here's the DataFrame my prof gave me.  I need to convert **Age** in Earth years to Martian years and then predict the person's favorite color.  **Give me code!**".  That's cheating, yo. 

* **When in doubt, ask**: If you have doubts about this policy or would like to discuss specific cases, please ask the instructor.

### Standard Course Policies 

**Honor Code**

All students enrolled in a University of Colorado Boulder course are responsible for knowing and adhering to the [academic integrity policy](http://www.colorado.edu/policies/academic-integrity-policy) of the institution. Violations of the policy may include: plagiarism, cheating, fabrication, lying, bribery, threat, unauthorized access, clicker fraud, resubmission, and aiding academic dishonesty. All incidents of academic misconduct will be reported to the Honor Code Council (honor@colorado.edu; 303-735-2273). Students who are found responsible for violating the academic integrity policy will be subject to nonacademic sanctions from the Honor Code Council as well as academic sanctions from the faculty member. Additional information regarding the academic integrity policy can be found at [www.colorado.edu/honorcode/](http://www.colorado.edu/honorcode/).

**Disability Accommodations**


If you qualify for accommodations because of a disability, please submit your accommodation letter from Disability Services to your faculty member in a timely manner (for exam accommodations provide your letter at least one week prior to the exam) so that your needs can be addressed.  Disability Services determines accommodations based on documented disabilities in the academic environment.  Information on requesting accommodations is located on the [Disability Services website](http://www.colorado.edu/disabilityservices/students).  Contact Disability Services at 303-492-8671 or [dsinfo@colorado.edu](dsinfo@colorado.edu) for further assistance.  If you have a temporary medical condition or injury, see Temporary Medical Conditions under the Students tab on the Disability Services website and discuss your needs with your professor.


**Religious Observances**

Campus policy regarding religious observances requires that faculty make every effort to deal reasonably and fairly with all students who, because of religious obligations, have conflicts with scheduled exams, assignments, or required attendance. If you have an exam or assignment conflict due to a religious observance please notify your instructor in a timely manner. See the [campus policy regarding religious observances](http://www.colorado.edu/policies/observance-religious-holidays-and-absences-classes-andor-exams) for full details.

**Classroom Behavior**

Students and faculty each have responsibility for maintaining an appropriate learning environment. Those who fail to adhere to such behavioral standards may be subject to discipline. Professional courtesy and sensitivity are especially important with respect to individuals and topics dealing with race, color, national origin, sex, pregnancy, age, disability, creed, religion, sexual orientation, gender identity, gender expression, veteran status, political affiliation or political philosophy.  Class rosters are provided to the instructor with the student's legal name. We will gladly honor your request to address you by an alternate name or gender pronoun. Please advise me of this preference early in the semester so that I may make appropriate changes to my records.  For more information, see the policies on [classroom behavior](http://www.colorado.edu/policies/student-classroom-and-course-related-behavior) and the [Student Code of Conduct](http://www.colorado.edu/osccr/).


**Sexual Misconduct, Discrimination, Harassment and/or Related Retaliation**

The University of Colorado Boulder (CU Boulder) is committed to maintaining a positive learning, working, and living environment. CU Boulder will not tolerate acts of sexual misconduct, discrimination, harassment or related retaliation against or by any employee or student. CU's Sexual Misconduct Policy prohibits sexual assault, sexual exploitation, sexual harassment, intimate partner abuse (dating or domestic violence), stalking or related retaliation. CU Boulder's Discrimination and Harassment Policy prohibits discrimination, harassment or related retaliation based on race, color, national origin, sex, pregnancy, age, disability, creed, religion, sexual orientation, gender identity, gender expression, veteran status, political affiliation or political philosophy. Individuals who believe they have been subject to misconduct under either policy should contact the Office of Institutional Equity and Compliance (OIEC) at 303-492-2127. Information about the OIEC, the above referenced policies, and the campus resources available to assist individuals regarding sexual misconduct, discrimination, harassment or related retaliation can be found at the [OIEC website](http://www.colorado.edu/institutionalequity/).



