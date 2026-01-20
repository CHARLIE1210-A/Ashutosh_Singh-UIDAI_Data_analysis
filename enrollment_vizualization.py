
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Mount Google Drive (if not already mounted)
# In a deployed Streamlit app, you might not need this
# drive.mount('/content/drive')

st.set_page_config(layout="wide")
st.title('Aadhaar Enrollment Data Analysis')

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    # Assuming Google Drive is already mounted or relevant files are accessible
    base_path = os.path.join(os.getcwd(), 'datasets')
    enrolment_subfolder_path = os.path.join(base_path, 'api_data_aadhar_enrolment')

    files_to_load = [
        'api_data_aadhar_enrolment_0_500000.csv',
        'api_data_aadhar_enrolment_500000_1000000.csv',
        'api_data_aadhar_enrolment_1000000_1006029.csv'
    ]

    df_list = []
    for file_name in files_to_load:
        file_path = os.path.join(enrolment_subfolder_path, file_name)
        try:
            df_list.append(pd.read_csv(file_path))
        except FileNotFoundError:
            st.error(f"Error: {file_name} not found. Please ensure Google Drive is mounted and the path is correct.")
            return pd.DataFrame() # Return empty if essential files are missing
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
            return pd.DataFrame() # Return empty if essential files are missing

    if df_list:
        df_enrolment = pd.concat(df_list, ignore_index=True)
    else:
        st.warning("No Aadhaar enrolment data was loaded.")
        return pd.DataFrame()
    return df_enrolment

df_enrolment = load_data()

if not df_enrolment.empty:
    st.header('1. Raw Data Preview')
    st.write(df_enrolment.head())

    # --- 2. Data Cleaning and Preprocessing ---
    df_enrolment['date'] = pd.to_datetime(df_enrolment['date'], format='%d-%m-%Y')
    df_enrolment.drop_duplicates(inplace=True)
    df_enrolment.reset_index(drop=True, inplace=True)

    # Standardize 'state' column
    df_enrolment['state'] = df_enrolment['state'].str.replace(r'West\s*Bengal|Westbengal|WEST BENGAL|West Bangal', 'West Bengal', regex=True)
    df_enrolment['state'] = df_enrolment['state'].str.replace('Orissa', 'Odisha', regex=False)
    df_enrolment['state'] = df_enrolment['state'].str.replace(r'Jammu and Kashmir|Jammu & Kashmir|Jammu And Kashmir', 'Jammu and Kashmir', regex=True)
    df_enrolment['state'] = df_enrolment['state'].str.replace(r'Dadra and Nagar Haveli and Daman and Diu|Dadra and Nagar Haveli|Dadra & Nagar Haveli|Daman and Diu|Daman & Diu', 'Dadra and Nagar Haveli and Daman and Diu', regex=True)
    df_enrolment['state'] = df_enrolment['state'].str.replace('andhra pradesh', 'Andhra Pradesh', regex=False)
    df_enrolment.loc[df_enrolment['state'] == '100000', 'state'] = 'Unknown'

    st.header('2. Data Cleaning Summary')
    st.write(f"Number of duplicate rows removed: {df_enrolment.duplicated().sum()} (after removing)")
    st.write("Missing values:", df_enrolment.isnull().sum()[df_enrolment.isnull().sum() > 0])

    # --- 3. Feature Engineering ---
    df_enrolment['total_enrollments'] = df_enrolment['age_0_5'] + df_enrolment['age_5_17'] + df_enrolment['age_18_greater']
    df_enrolment['year'] = df_enrolment['date'].dt.year
    df_enrolment['month'] = df_enrolment['date'].dt.month
    df_enrolment['prop_age_0_5'] = df_enrolment['age_0_5'] / df_enrolment['total_enrollments']
    df_enrolment['prop_age_5_17'] = df_enrolment['age_5_17'] / df_enrolment['total_enrollments']
    df_enrolment['prop_age_18_greater'] = df_enrolment['age_18_greater'] / df_enrolment['total_enrollments']
    df_enrolment.fillna(0, inplace=True)

    st.header('3. Feature Engineering Summary')
    st.write("New columns 'total_enrollments', 'year', 'month', and age proportion columns created.")
    st.write(df_enrolment[['date', 'total_enrollments', 'year', 'month', 'prop_age_0_5', 'prop_age_5_17', 'prop_age_18_greater']].head())

    # --- 4. Descriptive Statistics and Key Metrics ---
    st.header('4. Descriptive Statistics and Key Metrics')
    st.subheader('Overall DataFrame Info')
    # st.write(df_enrolment.info()) # Streamlit doesn't display .info() directly, use description
    st.write(df_enrolment.describe(include='all'))

    st.subheader('Descriptive statistics for numerical columns')
    st.write(df_enrolment.describe())

    st.subheader('Descriptive statistics for age-group enrollment counts')
    st.write(df_enrolment[['age_0_5', 'age_5_17', 'age_18_greater']].describe())

    enrollments_by_state = df_enrolment.groupby('state')['total_enrollments'].sum().sort_values(ascending=False)
    enrollments_by_district = df_enrolment.groupby('district')['total_enrollments'].sum().sort_values(ascending=False)
    enrollments_by_pincode = df_enrolment.groupby('pincode')['total_enrollments'].sum().sort_values(ascending=False)

    st.subheader('Top 10 States by Total Enrollments')
    st.write(enrollments_by_state.head(10))

    st.subheader('Top 10 Districts by Total Enrollments')
    st.write(enrollments_by_district.head(10))

    st.subheader('Top 10 Pincodes by Total Enrollments')
    st.write(enrollments_by_pincode.head(10))

    enrollments_over_time = df_enrolment.groupby(['year', 'month'])['total_enrollments'].sum().reset_index()
    st.subheader('Total enrollments over time (by year and month)')
    st.write(enrollments_over_time.head())

    total_age_0_5 = df_enrolment['age_0_5'].sum()
    total_age_5_17 = df_enrolment['age_5_17'].sum()
    total_age_18_greater = df_enrolment['age_18_greater'].sum()
    overall_total_enrollments = total_age_0_5 + total_age_5_17 + total_age_18_greater

    percentage_age_0_5 = (total_age_0_5 / overall_total_enrollments) * 100
    percentage_age_5_17 = (total_age_5_17 / overall_total_enrollments) * 100
    percentage_age_18_greater = (total_age_18_greater / overall_total_enrollments) * 100

    st.subheader('Total Enrollments by Age Group')
    st.write(f"Total enrollments for age group 0-5: {total_age_0_5} ({percentage_age_0_5:.2f}%) ")
    st.write(f"Total enrollments for age group 5-17: {total_age_5_17} ({percentage_age_5_17:.2f}%) ")
    st.write(f"Total enrollments for age group 18-greater: {total_age_18_greater} ({percentage_age_18_greater:.2f}%) ")

    overall_total_enrollment_districts = enrollments_by_district.sum()
    num_unique_districts = enrollments_by_district.shape[0]
    n_districts = max(1, int(num_unique_districts * 0.10))
    top_n_districts_enrollments_sum = enrollments_by_district.head(n_districts).sum()
    district_concentration_percentage = (top_n_districts_enrollments_sum / overall_total_enrollment_districts) * 100
    st.write(f"District Concentration Index (Top {n_districts} Districts): {district_concentration_percentage:.2f}%")

    overall_total_enrollment_pincodes = enrollments_by_pincode.sum()
    num_unique_pincodes = enrollments_by_pincode.shape[0]
    n_pincodes = max(1, int(num_unique_pincodes * 0.10))
    top_n_pincodes_enrollments_sum = enrollments_by_pincode.head(n_pincodes).sum()
    pincode_concentration_percentage = (top_n_pincodes_enrollments_sum / overall_total_enrollment_pincodes) * 100
    st.write(f"Pincode Concentration Index (Top {n_pincodes} Pincodes): {pincode_concentration_percentage:.2f}%")

    # --- 5. Visualizations ---
    st.header('5. Visualizations')

    st.subheader('5.1 Top 10 States by Aadhaar Enrollments')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=enrollments_by_state.head(10).index, y=enrollments_by_state.head(10).values, hue=enrollments_by_state.head(10).index, palette='viridis', ax=ax, legend=False)
    ax.set_title('Top 10 States by Aadhaar Enrollments')
    ax.set_xlabel('State')
    ax.set_ylabel('Total Enrollments')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.subheader('5.2 Top 10 Districts by Aadhaar Enrollments')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=enrollments_by_district.head(10).index, y=enrollments_by_district.head(10).values, hue=enrollments_by_district.head(10).index, palette='viridis', ax=ax, legend=False)
    ax.set_title('Top 10 Districts by Aadhaar Enrollments')
    ax.set_xlabel('District')
    ax.set_ylabel('Total Enrollments')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.subheader('5.3 Top 10 Pincodes by Aadhaar Enrollments')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=enrollments_by_pincode.head(10).index.astype(str), y=enrollments_by_pincode.head(10).values, hue=enrollments_by_pincode.head(10).index.astype(str), palette='viridis', ax=ax, legend=False)
    ax.set_title('Top 10 Pincodes by Aadhaar Enrollments')
    ax.set_xlabel('Pincode')
    ax.set_ylabel('Total Enrollments')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.subheader('5.4 Aadhaar Enrollments Over Time')
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=enrollments_over_time, x=enrollments_over_time['year'].astype(str) + '-' + enrollments_over_time['month'].astype(str), y='total_enrollments', marker='o', ax=ax)
    ax.set_title('Aadhaar Enrollments Over Time')
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Total Enrollments')
    ax.tick_params(axis='x', rotation=60)
    ax.grid(True)
    st.pyplot(fig)

    st.subheader('5.5 Total Aadhaar Enrollments by Age Group')
    age_group_enrollments = pd.DataFrame({
        'Age Group': ['0-5', '5-17', '18+'],
        'Total Enrollments': [total_age_0_5, total_age_5_17, total_age_18_greater]
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Age Group', y='Total Enrollments', data=age_group_enrollments, hue='Age Group', palette='viridis', ax=ax, legend=False)
    ax.set_title('Total Aadhaar Enrollments by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Total Enrollments')
    st.pyplot(fig)

    st.subheader('5.6 Distribution of Proportion of Enrollments by Age Group')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df_enrolment['prop_age_0_5'], bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Distribution of Proportion of Enrollments (Age 0-5)')
    axes[0].set_xlabel('Proportion')
    axes[0].set_ylabel('Frequency')

    sns.histplot(df_enrolment['prop_age_5_17'], bins=30, kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Proportion of Enrollments (Age 5-17)')
    axes[1].set_xlabel('Proportion')
    axes[1].set_ylabel('Frequency')

    sns.histplot(df_enrolment['prop_age_18_greater'], bins=30, kde=True, ax=axes[2])
    axes[2].set_title('Distribution of Proportion of Enrollments (Age 18+)')
    axes[2].set_xlabel('Proportion')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('5.7 Comparison of Concentration Indices')
    concentration_data = {
        'Category': [f'District (Top {n_districts})', 'Age Group (0-5)'],
        'Percentage': [district_concentration_percentage, percentage_age_0_5]
    }
    df_concentration = pd.DataFrame(concentration_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Category', y='Percentage', data=df_concentration, hue='Category', palette='coolwarm', ax=ax, legend=False)
    ax.set_title('Comparison of Aadhaar Enrollment Concentration Indices')
    ax.set_xlabel('Concentration Index Type')
    ax.set_ylabel('Percentage of Total Enrollments')
    ax.set_ylim(0, 100)

    for index, row in df_concentration.iterrows():
        ax.text(index, row['Percentage'] + 2, f"{row['Percentage']:.2f}%", color='black', ha="center")

    st.pyplot(fig)

    st.subheader('5.8 Distribution of Total Aadhaar Enrollments')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_enrolment['total_enrollments'], bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Total Aadhaar Enrollments')
    ax.set_xlabel('Total Enrollments')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    
    # New visualization: Enrollments over time by age group
    st.subheader('5.9 Aadhaar Enrollments Over Time by Age Group')
    enrollments_by_age_over_time = df_enrolment.groupby(['year', 'month'])[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
    enrollments_by_age_over_time['YearMonth'] = enrollments_by_age_over_time['year'].astype(str) + '-' + enrollments_by_age_over_time['month'].astype(str)
    
    df_melted = enrollments_by_age_over_time.melt(id_vars=['YearMonth', 'year', 'month'], 
                                                    value_vars=['age_0_5', 'age_5_17', 'age_18_greater'], 
                                                    var_name='Age Group', 
                                                    value_name='Total Enrollments')
    
    # Map age group names for better readability
    age_group_map = {'age_0_5': '0-5 Years', 'age_5_17': '5-17 Years', 'age_18_greater': '18+ Years'}
    df_melted['Age Group'] = df_melted['Age Group'].map(age_group_map)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df_melted, x='YearMonth', y='Total Enrollments', hue='Age Group', marker='o', ax=ax)
    ax.set_title('Aadhaar Enrollments Over Time by Age Group')
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Total Enrollments')
    ax.tick_params(axis='x', rotation=60)
    ax.grid(True)
    st.pyplot(fig)



    st.header('6. Comprehensive Summary of Findings and KPIs')
    st.markdown("""
    The most important Key Performance Indicators (KPIs) for the Aadhaar enrollment data, their values, patterns, trends, and implications are as follows:

    *   **Total Aadhaar Enrollments**: The dataset contained a total of 5,331,760 Aadhaar enrollments, indicating the overall scale of the enrollment activity.
    *   **State-wise Enrollment Distribution**: Uttar Pradesh (1,002,631), Bihar (593,753), Madhya Pradesh (487,892), West Bengal (369,241), and Maharashtra (363,446) recorded the highest enrollments, suggesting significant activity concentrated in these populous regions.
    *   **District-wise Enrollment Distribution**: Thane (43,142), Sitamarhi (41,652), Bahraich (38,897), Murshidabad (34,968), and South 24 Parganas (33,088) were the leading districts, representing granular hotspots potentially due to high population density or effective local outreach.
    *   **District Concentration Index**: The top 98 districts contributed 38.69% of total enrollments, indicating a moderate level of geographical concentration and a relatively distributed enrollment pattern across the country.
    *   **Pincode Concentration Index**: The top 2,786 pincodes accounted for 57.70% of enrollments, highlighting highly localized hotspots within districts.
    *   **Age Group Enrollment Contribution**: The 0-5 years age group had the highest enrollments (3,474,389, or 65.16% of the total), followed by 5-17 years (1,690,909, or 31.71%), and 18+ years (166,462, or 3.12%). This indicates a strong focus on enrolling young children.
    *   **Enrollment Trends Over Time**: Monthly enrollment data for 2025 showed fluctuations, with peaks in July (616,868 enrollments) and September (1,475,879 enrollments), followed by a decline. This suggests variability in enrollment activity, possibly influenced by campaign timings or administrative pushes.
    *   **Proportion of Enrollments by Age Group**: While the 0-5 age group dominated overall, localized variations were observed, indicating a mixed demographic focus depending on specific enrollment events.

    ### Data Analysis Key Findings
    *   The total number of Aadhaar enrollments analyzed was 5,331,760, reflecting the overall scale of enrollment efforts.
    *   Enrollment activity is highly concentrated in specific states, with Uttar Pradesh, Bihar, Madhya Pradesh, West Bengal, and Maharashtra being the top 5 contributors.
    *   At a more granular level, districts like Thane, Sitamarhi, Bahraich, Murshidabad, and South 24 Parganas show significant enrollment hotspots.
    *   Geographical concentration is moderate at the district level (top 98 districts account for 38.69% of enrollments) but becomes more pronounced at the pincode level (top 2,786 pincodes account for 57.70% of enrollments).
    *   The 0-5 years age group accounts for the majority of enrollments, contributing 65.16% (3,474,389 enrollments), followed by the 5-17 years age group at 31.71% (1,690,909 enrollments), and the 18+ years age group at only 3.12% (166,462 enrollments).
    *   Enrollment trends show significant monthly fluctuations in 2025, with notable peaks in July and September, indicating that external factors or targeted campaigns might influence enrollment rates.

    """)
else:
    st.error("Unable to load data. Please ensure the dataset is available at the specified path.")
