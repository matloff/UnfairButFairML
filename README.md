
# When "Unfair" ML May Actually Be Fair

[Prof. Norman Matloff, UC Davis](http://heather.cs.ucdavis.edu/itaa.html )

Today's rapidly increasing interest in the fairness of machine learning
(ML) algorithms is easily explained using the "Hello World" of fair ML,
the COMPAS project.  A [**Pro Publica**
article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
investigated COMPAS, an ML algorithm designed to predict recidivism by
those convicted of crimes.  The article found the tool to be racially
biased, of major concern since it was being used by judges as an aid in
sentencing convicts.  The developers of the
algorithm later offered a rebuttal of the criticism, but in any case,
COMPAS well illustrates the basic issues.

## Goals of this note

Here, though, we revisit the basic issue of "fairness."  We also show
that the approach here can lead to more accurate ML predictions in
general, apart from the fairness issue.

## Notation

We are using a feature vector X to predict Y. The components of X
include a set S of sensitive variables such as race or gender, and
possibly a set C of variables that are not directly sensitive but are
correlated strongly enough with S to divulge information about S
even if S is omitted from the analysis.

## Should S be used in prediction?

As noted in
[Gaebler et al, 2023)](https://5harad.com/papers/fair-ml.pdf):

> ...the relationship between a predictor and an outcome can differ across
> groups— what Ayres (2002) calls the problem of subgroup
> validity—potentially skewing estimates that ignore such distinctions.
> For example, the incidence of diabetes varies across race groups, even
> after adjusting for age and body mass index (BMI) (Aggarwal et al.,
> 2022).  Consequently, diabetes risk calculators that do not adjust for
> these differences may lead to inequitable medical care. When labels are
> accurately measured, this phenomenon can be countered by fitting
> group-specific risk models that learn such idiosyncratic patterns. That
> approach, however, requires explicitly incorporating group membership
> into risk estimates, potential[ly] creating other legal and policy concerns.

This issue has been much discussed in the medical context, the focus of
our note here.  The authors in
[Vyas et al, 2020](https://www.nejm.org/doi/full/10.1056/NEJMms2004740)
illustrate some of the risks in using race in a predictive medical
algorithm:

> Cardiac surgeons also consider race. The Society of Thoracic Surgeons
> produces elaborate calculators to estimate the risk of death and other
> complications during surgery.10 The calculators include race and
> ethnicity because of observed differences in surgical outcomes among
> racial and ethnic groups; the authors acknowledge that the mechanism
> underlying these differences is not known. An isolated coronary artery
> bypass in a low-risk white patient carries an estimated risk of death of
> 0.492%. Changing the race to “black/African American” increases the risk
> by nearly 20%, to 0.586%. Changing to any other race or ethnicity does
> not increase the estimated risk of death as compared with a white
> patient, but it does change the risk of renal failure, stroke, or
> prolonged ventilation. When used preoperatively to assess risk, these
> calculations could steer minority patients, deemed to be at higher risk,
> away from surgery.

## But, is "unfair" ML necessarily harmful??

A criminal defendant who is unfairly given a longer sentence
due to his race is clearly harmed.  A patient who is unreasonably denied
needed surgery due to their race or gender is also harmed.  But let's take this second example further,
not just for surgery but also for risky treatments, expensive drugs and
so on.

**Doesn't the patient have a right to a FULL assessment of his/her risks of
undergoing surgery, consuming drugs with serious side effects and so
on?**  Again, the patient has a right to a FULL assessment, one that
takes into account race, gender or whatever.  Indeed, the Health
Insurance Portability and Accountability Act gives patients the right to
access their medical information, and arguably this extends to a
patient's right to have the most accurate medical diagnosis possible for
the patient's personal situation.  This may necessitate using the
patient's race and gender in the analysis.

The "legal and policy concerns" cited above by Gaebler *et al* are
motivated by a desire to *protect* individuals in a minority or oppressed
class.  But in our setting here, withholding the most accurate diagnosis
possible is making such a patient a *victim*, not protecting him/her.

In other words, so-called "unfair" ML, in taking race and gender into
account, **is actually FAIR** in this setting.  *Current definitions of
fairness are overly restrictive, bringing harm to all.*

We will return to this issue shortly.  But first, a crucial technical
point:

## Tradeoffs

Fair ML researchers speak of a Fairness-Utility Tradeoff, a term
inspired by the famous Bias-Variance Tradeoff.  It is seldom mentioned,
though, that the Bias-Variance Tradeoff has implications for the
Fairness-Utility Tradeoff.  Here's how:

* The Bias-Variance Tradeoff says that the *bias* of a
prediction--an inherent skewing that would persist even if the data set
were far larger--is at odds with the *variance*, the variability of the
prediction from one version of the dataset to another of the same size.
(E.g. in a diabetes study involving 1000 patients, we might think of
them as being sampled from the conceptual population of all diabetics,
and there are many possible datasets of 1000 such patients.)
More complex models have smaller bias but larger variance.  One
generally tries to find a "happy medium" betwen the two.

    In our context here, mocels that use S and C are more complex than
    those that don't.  (Unfairness mitigation algorithms, that use S and C
    but in reduced form, are in the middle.)

* The Fairness/Utility Tradeoff notes that fairness is at odds with
"utility," meaning predictive power.  The greater the degree to which S
and C are suppressed in an analyis, i.e. the fairer, the worse the
prediction accuracy.

The Bias-Variance is relevant in our setting here, as it affects
Utility.  Some implications:

* Even if S and C could be used in a fair manner, e.g. in the settings
  discussed above, it may not be desirable to do so.  The reduction in
bias may not be large enough to compensate for the increase in variance,
thus reducing Utility.

* On the other hand, omitting or reducing the use of S and C may results
  in a substantial increase in bias, possibly dominating a small
reduction in variance, thus reducing Utility.

* This suggests a hybrid approach:  If S and C can be used fully and
  fairly, it may be the case that a hybrid approach is best, with S and
C being taken into account for some S classes (e.g. women) but not
others (e.g. men).

## Putting this all together: an example dataset

Here we investigate these ideas on the 
[Early Stage Diabetes Risk Prediction Dataset](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.).  The data (with some cleaning) and the code are available in 
[my Github repo](https://github.com/matloff/UnfairButFairML).
The code uses my [qeML](https://github.com/matloff/qeML)
package ("Quick and Easy Machine Learning"), and is fairly general.  The
user is urged to try it on various datasets.  The example here is just a
beginning.

A simple logistic model was used, to predict the diabetic outcome, named
**class**.  For S, we simply used the **Gender** column.  There were no
C variables in this analysis (more on this shortly).  250 runs were
performed, with a random test set of size 10% of the total in each run.

Note that women were the minority in this dataset, with a 37%-63% gender
split.

Here the main results:

* In a female-only analysis, there was a misclassification rate of
  0.0530.

* In a male-only analysis, there was a misclassification rate of
  0.0811.

* In an analysis using the full dataset but not using S, the
  female and male misclassification rates were 0.1403 and 0.0959,
  respectively.

* In an analysis using the full dataset including S, the
  female and male misclassification rates were 0.0552 and 0.0728,
  respectively.

So, the "traditional" approach, in which S is excluded, did
substantially worse among women, compared to the class-specific
analyses.  Men were largely unaffected.

## Discussion of the fair ML issue

* We did not try any bias mitigation methods, which could possibly have
  reduced the gap between using or not using S (though possibly
  increasing sampling variability).

* We did not investigate possible C features; we simply included
  everything.  Note, though, that potentially, exclusion of C variables,
  or weakening their impact via mitigation methodology, may exacerbate
  the gap between "fair" analyses and those making use of all variables.

* Again, if the patient wants the most accurate diagnosis and is willing
  to have her gender used in that analysis, she should have a right to
  it.  Depriving her of the most accurate analysis would have given her
  a much less accurate result.

* In some applications, some S-classes may be so small that performing
  an analysis for those specific classes may be on the wrong side of the
  Bias-Variance Tradeoff.  Cross-validation, informed by medical domain
  expertise, may be used to evaluate whether a class-specific or overall
  anaysis is better for a particular class.

## General usefulness, outside the fair ML realm

