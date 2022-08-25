# Credit-Card-Default-Predicition

iNeuron.ai Data Science Internship Project


# Overview
This is a classification model for a most common dataset, Credit Card defaulter prediction. Prediction of the next month credit card defaulter based on demographic and last six months behavioral data of customers.

# Motivation
There are times when even a seemingly manageable debt, such as credit cards, goes out of control. Loss of job, medical crisis or business failure are some of the reasons that can impact your finances. In fact, credit card debts are usually the first to get out of hand in such situations due to hefty finance charges (compounded on daily balances) and other penalties.

A lot of us would be able to relate to this scenario. We may have missed credit card payments once or twice because of forgotten due dates or cash flow issues. But what happens when this continues for months? How to predict if a customer will be defaulter in next months?

To reduce the risk of Banks, this model has been developed to predict customer defaulter based on demographic data like gender, age, marital status and behavioral data like last payments, past transactions etc.

# Dataset Information
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

# Technical Aspect
This project is divided into two part:
<ol dir="auto">
<li>Training a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html" rel="nofollow">RandomForestClassifier</a> classification model to predict defaulter as accurate as possible.
<ul dir="auto">
<li>Cleaning the datasets, fixing all features</li>
<li>Apply Classification ML model</li>
</ul>
</li>
<li>Building and hosting a Flask web app on Heroku.
<ul dir="auto">
<li>Build the web app using Flask API</li>
<li>Upload the project on GitHub</li>
<li>Get the customer information from Web app</li>
<li>Display the prediction</li>
</ul>
</li>
</ol>

# Installation

The Code is written in Python 3.7. If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:

<div class="highlight highlight-source-shell notranslate position-relative overflow-auto"><pre>pip install -r requirements.txt</pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="pip install -r requirements.txt" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon m-2">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
  
  
  
  # Directory Tree
  
  <div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>├── templates 
│&nbsp;&nbsp; └── index.html
├── app.py
├── credit-card-default.csv
├── credit_default_prediction.py
├── model.pkl
├── Procfile
├── README.md
├── HLD document
├── LLD Document
├── Detailed Description Presentation
├── log file
├── wireframe pdf
├── README.md
└── requirements.txt
</code></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="├── templates 
│&nbsp;&nbsp; └── index.html
├── app.py
├── credit-card-default.csv
├── credit_default_prediction.py
├── model.pkl
├── Procfile
├── README.md
├── HLD document
├── LLD Document
├── Detailed Description Presentation
├── log file
├── wireframe pdf
├── README.md
└── requirements.txt" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon m-2">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>

# Technologies Used

<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://camo.githubusercontent.com/3cdf9577401a2c7dceac655bbd37fb2f3ee273a457bf1f2169c602fb80ca56f8/68747470733a2f2f666f7274686562616467652e636f6d2f696d616765732f6261646765732f6d6164652d776974682d707974686f6e2e737667"><img src="https://camo.githubusercontent.com/3cdf9577401a2c7dceac655bbd37fb2f3ee273a457bf1f2169c602fb80ca56f8/68747470733a2f2f666f7274686562616467652e636f6d2f696d616765732f6261646765732f6d6164652d776974682d707974686f6e2e737667" alt="" data-canonical-src="https://forthebadge.com/images/badges/made-with-python.svg" style="max-width: 100%;"></a></p>

<p dir="auto"><a href="https://numpy.org" rel="nofollow"><img src="https://camo.githubusercontent.com/c87d43dedad06f0c31855c1bc9f08a0e8b09e6f8998fecd1c051dc9ae51d75ac/68747470733a2f2f6e756d70792e6f72672f696d616765732f6c6f676f732f6e756d70792e737667" width="100" data-canonical-src="https://numpy.org/images/logos/numpy.svg" style="max-width: 100%;"></a>    <a href="https://pandas.pydata.org" rel="nofollow"><img src="https://camo.githubusercontent.com/7eb4beb24552e618fbdf738058c67fa29e50ecbcb628f77f503d73cd8a8c7db0/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f652f65642f50616e6461735f6c6f676f2e7376672f34353070782d50616e6461735f6c6f676f2e7376672e706e67" width="150" data-canonical-src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/450px-Pandas_logo.svg.png" style="max-width: 100%;"></a>    <a href="https://scikit-learn.org/stable" rel="nofollow"><img src="https://camo.githubusercontent.com/1d558c40dabf9c6ba6000aee6bf0831cbae21ee825097a26049f98757ba071fb/68747470733a2f2f7363696b69742d6c6561726e2e6f72672f737461626c652f5f7374617469632f7363696b69742d6c6561726e2d6c6f676f2d736d616c6c2e706e67" width="150" data-canonical-src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" style="max-width: 100%;"></a>   <a href="https://www.statsmodels.org" rel="nofollow"><img src="https://camo.githubusercontent.com/95f5613d9114673f0aef5195d9f732373692b1f3068978a1f93646d463928c00/68747470733a2f2f7777772e73746174736d6f64656c732e6f72672f737461626c652f5f696d616765732f73746174736d6f64656c732d6c6f676f2d76322d686f72697a6f6e74616c2e737667" width="170" data-canonical-src="https://www.statsmodels.org/stable/_images/statsmodels-logo-v2-horizontal.svg" style="max-width: 100%;"></a></p>

<p dir="auto"><a href="https://matplotlib.org" rel="nofollow"><img src="https://camo.githubusercontent.com/1b50dc4a1670e8748da0063c0728673f060eef77798141c326e55550ad7e1aea/68747470733a2f2f6d6174706c6f746c69622e6f72672f5f7374617469632f6c6f676f325f636f6d707265737365642e737667" width="170" data-canonical-src="https://matplotlib.org/_static/logo2_compressed.svg" style="max-width: 100%;"></a>      <a href="https://seaborn.pydata.org" rel="nofollow"><img src="https://camo.githubusercontent.com/bb7c2b8c732065da2b9d6cee6266ae2e07fb1188921e551948517bcb9e14503a/68747470733a2f2f736561626f726e2e7079646174612e6f72672f5f7374617469632f6c6f676f2d776964652d6c6967687462672e737667" width="150" data-canonical-src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" style="max-width: 100%;"></a></p>
