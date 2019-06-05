# Intro to Programming for Public Policy
# PPHA 30550


## Final Project: Reproducible Research
## Spring 2019


## Due: Monday, June 10th on GitHub Classrooms

### Instructions:
The goal of this project is to showcase your new-found knowledge of Python by applying it to a 
research project of your own.  This is not a research methods course, so the quality of the
research is ancillary to the quality of your programming.

You may work on this project *alone* or in groups of *no more than three people*.  If you choose
to work alone, you will use your own repository like normal.  If you choose to use a group, you
must create your own team, [according to the instructions here](https://github.blog/2018-03-06-how-to-use-group-assignments-in-github-classroom/)

It is required that you use GitHub, and I may use your past commits to understand your thought
process for partial credit, and to monitor group participation.  You will not get full credit if
your repository does not show multiple commits as you build your project, especially for groups.
Expectations for the quality, and especially the scope, of the code will be higher for groups 
than for individuals.  Work in a group only if you want to work with others, not if you want to 
reduce your workload.

------

### Project Description:
Your task is to find a minimum of two datasets available online, and then use Python to
download them using requests or Pandas, merge or concatenate them together, summarize the data 
with plots, tables and summary statistics, and then fit a simple model to it using Numpy or 
Statsmodels.  The entire program must be organized using functions and/or classes, and it must
run from start (downloading) to finish (making graphs, displaying tables, and model fitting) 
without stopping or requiring manual steps.

You will then spend 5 pages or less writing your research findings up.  This is where you can
discuss limitations as well.  For example, if you used an OLS model and got weak results, you
might discuss what model you would have used in future research, or what additional data you
would want to find, or why the OLS wasn't a good choice.  Think of this writeup as a guide to 
your code, as much as it is an explanation of your research.

Your repository must contain the following: 
1. Your code and commit history
2. The initial, unmodified dataframes you download
3. Saved .png versions of your plots
4. The final versions of the dataframe(s) you built
5. Your writeup (Word or markdown are both fine)

------

### Suggestions and Tips:
- There will be no more homework assignments this quarter, so you can focus on your project.
- We will learn about downloading data using Python this week.  In the mean time, look at the 
list of sources [Pandas Datareader](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html) can access to start thinking about your project.
- Using Python requests to download data with less structure is harder than using Datareader,
and will be taken into consideration when grading.  You can still get full credit using
Datareader, but plan to put more work into a different part of your project in that case.
- Anywhere you can use things we learned in class, in a way that is relevant to your project,
is a good idea.  For example, cleaning up your plots using some MatPlotLib methods we discussed
is better than turning in vanilla MatPlotLib output, using formatting and other functions to
clean up your data is better than displaying the same output in a messy fashion, and so on.
- Effort put into organizing your code and making it readable, by, for example, following the
Python Style Guide and good usage of functions and variable names, will be rewarded.
- Common merging and grouping parameters:
  - State
  - Country
  - Date
  - Names
- If you feel stuck coming up with research ideas, feel free to contact me or one of the TAs
so we can discuss your interests and make suggestions.
- If your research idea is a much larger project, think of how you can develop a basic framework
for it using this project, which can then later be expanded into a proper research project.
- Remember that the point here is to showcase your Python coding, so do not get hung up for too
long on questions of research methodology.
- The entire point of reproducible research is to make it possible for others (and for a future
you who has had time to forget what you did and why) to understand, replicate, and modify your
work.  Keeping this in mind as you work will be helpful 