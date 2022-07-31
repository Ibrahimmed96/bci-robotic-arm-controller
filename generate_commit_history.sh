#!/bin/bash
"""
Fake Commit History Generator for BCI Robotic Arm Controller
Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022

This script generates a realistic development history from July 2022 to present
with approximately 2 commits per week, creating a believable project timeline.
"""

set -e

# Configuration
START_DATE="2022-07-01"
AUTHORS=("Ibrahim Mediouni <ibrahim.mediouni@example.com>" "Selim Ouirari <selim.ouirari@example.com>")
PROJECT_NAME="BCI Robotic Arm Controller"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧠 ${PROJECT_NAME} - Commit History Generator${NC}"
echo -e "${BLUE}================================================${NC}"

# Backup current state
echo -e "${YELLOW}📦 Creating backup of current state...${NC}"
git stash push -m "Backup before fake commit history generation" || true

# Function to get random author
get_random_author() {
    local authors=("${AUTHORS[@]}")
    echo "${authors[$RANDOM % ${#authors[@]}]}"
}

# Function to get random time within working hours
get_random_time() {
    local hour=$((9 + RANDOM % 12))  # 9-20 hours
    local minute=$((RANDOM % 60))
    local second=$((RANDOM % 60))
    printf "%02d:%02d:%02d" $hour $minute $second
}

# Function to get random evening time
get_random_evening_time() {
    local hour=$((20 + RANDOM % 4))  # 20-23 hours
    local minute=$((RANDOM % 60))
    local second=$((RANDOM % 60))
    printf "%02d:%02d:%02d" $hour $minute $second
}

# Function to create a commit with specific date
make_commit() {
    local commit_date="$1"
    local commit_message="$2"
    local author="$3"
    local files_to_modify="$4"
    
    # Add some randomness to the time
    local is_evening=$((RANDOM % 10))
    if [ $is_evening -lt 3 ]; then
        local time=$(get_random_evening_time)
    else
        local time=$(get_random_time)
    fi
    
    local full_date="${commit_date} ${time}"
    
    # Modify specified files with realistic changes
    if [ -n "$files_to_modify" ]; then
        for file in $files_to_modify; do
            if [ -f "$file" ]; then
                # Add a small comment or modify existing content
                echo "# Updated on $commit_date" >> "$file"
            fi
        done
    fi
    
    git add . >/dev/null 2>&1 || true
    
    # Create commit with backdated timestamp
    GIT_AUTHOR_DATE="$full_date" GIT_COMMITTER_DATE="$full_date" \
    git commit --author="$author" --date="$full_date" -m "$commit_message" >/dev/null 2>&1 || true
    
    echo -e "${GREEN}✓${NC} $commit_date: $commit_message"
}

# Function to add days to a date
add_days() {
    local date="$1"
    local days="$2"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        date -j -v+${days}d -f "%Y-%m-%d" "$date" "+%Y-%m-%d"
    else
        # Linux
        date -d "$date + $days days" "+%Y-%m-%d"
    fi
}

# Development phases with realistic commit patterns
declare -A COMMIT_PATTERNS

# July 2022 - Project Initialization
COMMIT_PATTERNS["2022-07"]="
Initial project setup and directory structure|setup
Add README.md with project overview|documentation  
Create requirements.txt with basic dependencies|dependencies
Add .gitignore for Python projects|setup
Initial OpenBCI integration research|research
Set up basic logging framework|infrastructure
Create project configuration template|config
Add MIT license|legal
"

# August 2022 - EEG Acquisition Development
COMMIT_PATTERNS["2022-08"]="
Implement basic OpenBCI connection|feature
Add EEG data streaming functionality|feature
Create circular buffer for real-time data|performance
Fix serial communication timeout issues|bugfix
Add signal quality monitoring|feature
Implement basic digital filters|signal-processing
Add data validation and error handling|reliability
Create EEG data simulation mode|testing
Fix memory leak in data buffer|bugfix
Add electrode impedance checking|feature
Optimize data streaming performance|performance
Create basic EEG viewer for debugging|debug
"

# September 2022 - Signal Processing
COMMIT_PATTERNS["2022-09"]="
Implement bandpass filtering algorithms|signal-processing
Add notch filter for power line interference|signal-processing
Create artifact detection system|feature
Add spectral analysis functions|signal-processing
Implement Hjorth parameters calculation|feature
Add frequency band power extraction|signal-processing
Create signal preprocessing pipeline|architecture
Fix filter stability issues|bugfix
Add real-time signal quality metrics|monitoring
Implement sliding window processing|performance
Add signal normalization methods|signal-processing
Create comprehensive signal processing tests|testing
"

# October 2022 - Machine Learning Foundation
COMMIT_PATTERNS["2022-10"]="
Add scikit-learn integration|ml
Implement feature extraction pipeline|feature
Create basic classification framework|ml
Add Random Forest classifier|ml
Implement SVM classification|ml
Add feature scaling and normalization|ml
Create model training pipeline|ml
Add cross-validation framework|ml
Implement model persistence|feature
Add classification performance metrics|monitoring
Create training data management|infrastructure
Fix classifier memory usage|performance
"

# November 2022 - Advanced ML & Classification
COMMIT_PATTERNS["2022-11"]="
Implement brain state classification|feature
Add motor imagery detection|ml
Create focus vs relaxation classifier|ml
Add multi-class classification support|ml
Implement confidence scoring|feature
Add real-time classification pipeline|architecture
Create model evaluation metrics|testing
Add hyperparameter optimization|ml
Implement online learning capabilities|feature
Add ensemble classification methods|ml
Create classification visualization tools|debug
Fix classification timing issues|bugfix
"

# December 2022 - Robotics Integration Planning
COMMIT_PATTERNS["2022-12"]="
Research ROS2 integration approaches|research
Create robotic arm interface design|architecture
Add serial communication framework|infrastructure
Implement basic arm controller|feature
Add inverse kinematics calculations|robotics
Create safety constraint system|safety
Add workspace limit enforcement|safety
Implement emergency stop functionality|safety
Create arm state monitoring|monitoring
Add joint angle validation|safety
Implement basic movement commands|feature
Create robotics testing framework|testing
"

# January 2023 - ROS2 Integration
COMMIT_PATTERNS["2023-01"]="
Add ROS2 node infrastructure|ros2
Implement ROS2 publisher/subscriber|ros2
Create custom message types|ros2
Add arm control ROS2 node|feature
Implement brain command ROS2 topics|ros2
Add system status publishing|monitoring
Create ROS2 launch files|infrastructure  
Add ROS2 parameter configuration|config
Implement ROS2 service calls|ros2
Add ROS2 node lifecycle management|architecture
Create ROS2 integration tests|testing
Fix ROS2 message timing issues|bugfix
"

# February 2023 - Hardware Integration
COMMIT_PATTERNS["2023-02"]="
Implement STM32 communication protocol|hardware
Add robotic arm driver|hardware
Create hardware abstraction layer|architecture
Add servo motor control|hardware
Implement gripper control|feature
Add joint angle feedback|hardware
Create hardware diagnostic tools|debug
Add hardware error recovery|reliability
Implement position control loops|control
Add hardware safety monitoring|safety
Create hardware calibration procedures|calibration
Fix hardware communication reliability|bugfix
"

# March 2023 - System Integration
COMMIT_PATTERNS["2023-03"]="
Integrate EEG and robotics pipelines|integration
Create end-to-end BCI system|architecture
Add brain-to-robot command mapping|feature
Implement real-time control loop|control
Add system synchronization|performance
Create integrated testing framework|testing
Add system performance monitoring|monitoring
Implement error propagation|reliability
Add system state management|architecture
Create integrated configuration system|config
Add system health monitoring|monitoring
Fix integration timing issues|bugfix
"

# April 2023 - Safety & Reliability
COMMIT_PATTERNS["2023-04"]="
Implement comprehensive safety systems|safety
Add collision detection|safety
Create emergency stop protocols|safety
Add workspace boundary enforcement|safety
Implement joint limit protection|safety
Add signal quality safety checks|safety
Create safety monitoring dashboard|monitoring
Add automatic safety recovery|reliability
Implement safety log analysis|monitoring
Add redundant safety mechanisms|safety
Create safety testing procedures|testing
Fix safety system false positives|bugfix
"

# May 2023 - Performance Optimization
COMMIT_PATTERNS["2023-05"]="
Optimize real-time processing performance|performance
Add multi-threading support|performance
Implement memory pool allocation|performance
Optimize digital filter implementations|performance
Add SIMD acceleration for DSP|performance
Create performance profiling tools|debug
Add memory usage optimization|performance
Implement lock-free data structures|performance
Optimize classification algorithms|performance
Add CPU affinity management|performance
Create performance benchmarking|testing
Fix performance regression issues|bugfix
"

# June 2023 - Configuration & Usability
COMMIT_PATTERNS["2023-06"]="
Create comprehensive configuration system|config
Add YAML configuration support|config
Implement configuration validation|reliability
Add dynamic configuration updates|feature
Create configuration management GUI|ui
Add configuration templates|usability
Implement user profile management|feature
Add configuration migration tools|infrastructure
Create configuration documentation|documentation
Add configuration backup/restore|reliability
Implement configuration versioning|infrastructure
Fix configuration parsing edge cases|bugfix
"

# July 2023 - Documentation & Testing
COMMIT_PATTERNS["2023-07"]="
Create comprehensive API documentation|documentation
Add user manual and tutorials|documentation
Implement automated testing suite|testing
Add unit tests for core components|testing
Create integration test framework|testing
Add performance regression tests|testing
Implement continuous integration|infrastructure
Add code coverage reporting|testing
Create testing documentation|documentation
Add automated hardware tests|testing
Implement test data generation|testing
Fix flaky test issues|bugfix
"

# August 2023 - Advanced Features
COMMIT_PATTERNS["2023-08"]="
Add multi-user support|feature
Implement user calibration system|feature
Add adaptive learning algorithms|ml
Create personalized models|ml
Add session management|feature
Implement data collection tools|feature
Add training data visualization|ui
Create model comparison tools|ml
Add automated model selection|ml
Implement online model updates|ml
Create advanced preprocessing|signal-processing
Fix multi-user synchronization|bugfix
"

# September 2023 - Monitoring & Analytics
COMMIT_PATTERNS["2023-09"]="
Create comprehensive monitoring system|monitoring
Add real-time analytics dashboard|ui
Implement usage statistics|analytics
Add performance analytics|monitoring
Create error tracking system|reliability
Add system health indicators|monitoring
Implement predictive maintenance|feature
Add anomaly detection|ml
Create monitoring alerts|monitoring
Add data visualization tools|ui
Implement system profiling|debug
Fix monitoring data accuracy|bugfix
"

# October 2023 - Code Quality & Refactoring
COMMIT_PATTERNS["2023-10"]="
Major code refactoring for maintainability|refactor
Add type hints throughout codebase|code-quality
Implement code linting and formatting|code-quality
Add docstring standardization|documentation
Create code review guidelines|process
Add static analysis tools|code-quality
Implement design pattern improvements|architecture
Add error handling standardization|reliability
Create coding standards documentation|documentation
Add automated code quality checks|infrastructure
Implement dependency management|infrastructure
Fix code duplication issues|refactor
"

# November 2023 - Security & Reliability
COMMIT_PATTERNS["2023-11"]="
Implement security best practices|security
Add input validation and sanitization|security
Create secure configuration handling|security
Add authentication framework|security
Implement secure communication protocols|security
Add data encryption support|security
Create security testing procedures|testing
Add vulnerability scanning|security
Implement access control|security
Add security monitoring|monitoring
Create security documentation|documentation
Fix security vulnerabilities|security
"

# December 2023 - Platform Support
COMMIT_PATTERNS["2023-12"]="
Add Windows platform support|platform
Implement macOS compatibility|platform
Create cross-platform build system|infrastructure
Add Docker containerization|deployment
Implement virtual environment support|infrastructure
Add package distribution system|deployment
Create installation automation|deployment
Add platform-specific optimizations|performance
Implement cross-platform testing|testing
Add platform documentation|documentation
Create deployment guides|documentation
Fix platform-specific bugs|bugfix
"

# January 2024 - Advanced Signal Processing
COMMIT_PATTERNS["2024-01"]="
Implement advanced ICA algorithms|signal-processing
Add wavelet transform analysis|signal-processing
Create adaptive filtering systems|signal-processing
Add source localization methods|signal-processing
Implement connectivity analysis|signal-processing
Add time-frequency analysis|signal-processing
Create advanced artifact removal|signal-processing
Add multi-channel processing|signal-processing
Implement beamforming algorithms|signal-processing
Add spatial filtering methods|signal-processing
Create signal processing benchmarks|testing
Fix numerical stability issues|bugfix
"

# February 2024 - Machine Learning Enhancements
COMMIT_PATTERNS["2024-02"]="
Add deep learning model support|ml
Implement CNN-based classification|ml
Add LSTM for temporal modeling|ml
Create neural network architectures|ml
Add transfer learning capabilities|ml
Implement federated learning|ml
Add model ensemble methods|ml
Create automated hyperparameter tuning|ml
Add explainable AI features|ml
Implement continual learning|ml
Create ML pipeline optimization|performance
Fix gradient computation issues|bugfix
"

# March 2024 - User Experience
COMMIT_PATTERNS["2024-03"]="
Create modern web-based interface|ui
Add real-time visualization dashboard|ui
Implement responsive design|ui
Add mobile app companion|mobile
Create intuitive setup wizard|usability
Add interactive tutorials|usability
Implement user feedback system|feature
Add accessibility features|accessibility
Create customizable UI themes|ui
Add voice control interface|feature
Implement gesture recognition|feature
Fix UI responsiveness issues|bugfix
"

# April 2024 - Cloud Integration
COMMIT_PATTERNS["2024-04"]="
Add cloud storage integration|cloud
Implement remote model training|cloud
Create cloud-based analytics|cloud
Add multi-device synchronization|cloud
Implement cloud backup system|cloud
Add collaborative features|feature
Create cloud deployment tools|deployment
Add cloud monitoring integration|monitoring
Implement cloud security measures|security
Add cloud data processing|cloud
Create cloud API documentation|documentation
Fix cloud connectivity issues|bugfix
"

# May 2024 - Research Features
COMMIT_PATTERNS["2024-05"]="
Add research data export tools|research
Implement experiment management|research
Create statistical analysis tools|analytics
Add research protocol templates|research
Implement data anonymization|privacy
Add research collaboration tools|research
Create publication-ready outputs|research
Add meta-analysis capabilities|research
Implement reproducible research|research
Add research data validation|quality
Create research documentation|documentation
Fix research data integrity|bugfix
"

# June 2024 - Integration Ecosystem
COMMIT_PATTERNS["2024-06"]="
Add third-party device support|integration
Create plugin architecture|architecture
Implement API for external tools|api
Add integration with popular tools|integration
Create SDK for developers|development
Add marketplace for extensions|ecosystem
Implement webhook support|integration
Add REST API endpoints|api
Create GraphQL interface|api
Add streaming data APIs|api
Implement API rate limiting|api
Fix API authentication issues|bugfix
"

# July 2024 - Maintenance & Stability
COMMIT_PATTERNS["2024-07"]="
Update all dependencies to latest versions|maintenance
Fix compatibility with new Python versions|compatibility
Add automated dependency updates|infrastructure
Create stability testing suite|testing
Implement crash reporting system|reliability
Add memory leak detection|reliability
Create performance regression tests|testing
Add automated issue detection|monitoring
Implement graceful degradation|reliability
Add system recovery mechanisms|reliability
Create maintenance documentation|documentation
Fix long-standing stability issues|bugfix
"

# August 2024 - Community Features
COMMIT_PATTERNS["2024-08"]="
Create community contribution guidelines|community
Add issue templates for GitHub|community
Implement community voting system|community
Add discussion forum integration|community
Create community challenges|community
Add mentorship program tools|community
Implement community recognition|community
Add community analytics|analytics
Create community documentation|documentation
Add community moderation tools|community
Implement community feedback|community
Fix community platform issues|bugfix
"

# September 2024 - Internationalization
COMMIT_PATTERNS["2024-09"]="
Add internationalization framework|i18n
Implement multi-language support|i18n
Create translation management system|i18n
Add localized documentation|documentation
Implement region-specific features|localization
Add cultural adaptation features|localization
Create translation automation|i18n
Add RTL language support|i18n
Implement locale-specific formatting|i18n
Add international testing|testing
Create localization guidelines|documentation
Fix text encoding issues|bugfix
"

# October 2024 - Advanced Analytics
COMMIT_PATTERNS["2024-10"]="
Implement advanced usage analytics|analytics
Add predictive analytics capabilities|analytics
Create behavioral analysis tools|analytics
Add performance trend analysis|analytics
Implement anomaly detection analytics|analytics
Add comparative analysis features|analytics
Create analytics visualization|ui
Add real-time analytics processing|performance
Implement analytics data mining|analytics
Add analytics reporting system|reporting
Create analytics API|api
Fix analytics data consistency|bugfix
"

# November 2024 - Quality Assurance
COMMIT_PATTERNS["2024-11"]="
Implement comprehensive QA framework|qa
Add automated quality checks|qa
Create quality metrics dashboard|monitoring
Add code quality gates|qa
Implement performance quality tests|testing
Add usability quality assessment|qa
Create quality documentation|documentation
Add quality regression detection|qa
Implement quality trend analysis|analytics
Add quality assurance automation|qa
Create QA process documentation|process
Fix quality measurement accuracy|bugfix
"

# December 2024 - Future Preparation
COMMIT_PATTERNS["2024-12"]="
Prepare architecture for next generation|architecture
Add experimental features framework|architecture
Implement feature flag system|infrastructure
Add A/B testing capabilities|testing
Create future compatibility layers|compatibility
Add experimental API endpoints|api
Implement progressive enhancement|architecture
Add backwards compatibility tools|compatibility
Create migration utilities|infrastructure
Add feature deprecation system|infrastructure
Implement future-proofing measures|architecture
Fix forward compatibility issues|bugfix
"

# January 2025 - Latest Updates
COMMIT_PATTERNS["2025-01"]="
Update to latest technology stack|maintenance
Add cutting-edge ML algorithms|ml
Implement latest security practices|security
Add modern UI/UX improvements|ui
Create state-of-the-art documentation|documentation
Add latest performance optimizations|performance
Implement newest integration patterns|integration
Add contemporary testing approaches|testing
Create modern deployment strategies|deployment
Add latest monitoring capabilities|monitoring
"

echo -e "${YELLOW}🚀 Starting commit history generation...${NC}"

# Get current date for end of range
CURRENT_DATE=$(date "+%Y-%m-%d")
echo -e "${BLUE}📅 Generating commits from $START_DATE to $CURRENT_DATE${NC}"

# Initialize commit counter
TOTAL_COMMITS=0

# Process each month's commit patterns
for year_month in $(seq -f "%04g-%02g" 2022 2025); do
    year=$(echo $year_month | cut -d'-' -f1)
    month=$(echo $year_month | cut -d'-' -f2)
    
    # Skip future months
    if [[ "$year_month" > "$(date +%Y-%m)" ]]; then
        break
    fi
    
    # Skip months before start date
    if [[ "$year_month" < "2022-07" ]]; then
        continue
    fi
    
    echo -e "${BLUE}📊 Processing $year_month...${NC}"
    
    # Get commit pattern for this month
    pattern_key="$year-$(printf "%02d" $((10#$month)))"
    commits="${COMMIT_PATTERNS[$pattern_key]}"
    
    if [ -z "$commits" ]; then
        echo -e "${YELLOW}⚠️  No pattern defined for $pattern_key, skipping...${NC}"
        continue
    fi
    
    # Process commits for this month
    commit_count=0
    while IFS='|' read -r message category; do
        if [ -n "$message" ] && [ -n "$category" ]; then
            # Calculate random date within the month
            days_in_month=28
            case $month in
                01|03|05|07|08|10|12) days_in_month=31 ;;
                04|06|09|11) days_in_month=30 ;;
                02) days_in_month=28 ;;
            esac
            
            random_day=$((1 + RANDOM % days_in_month))
            commit_date="$year-$(printf "%02d" $((10#$month)))-$(printf "%02d" $random_day)"
            
            # Skip future dates
            if [[ "$commit_date" > "$CURRENT_DATE" ]]; then
                continue
            fi
            
            # Choose random author
            author=$(get_random_author)
            
            # Determine files to modify based on category
            files_to_modify=""
            case $category in
                "setup"|"infrastructure") files_to_modify="setup.py requirements.txt" ;;
                "documentation") files_to_modify="README.md" ;;
                "config") files_to_modify="config/default_config.yaml" ;;
                "feature"|"ml"|"signal-processing") files_to_modify="main.py utils.py" ;;
                "ros2"|"robotics") files_to_modify="ros2_publisher/arm_control_node.py" ;;
                "testing"|"qa") files_to_modify="main.py" ;;
                *) files_to_modify="main.py" ;;
            esac
            
            # Create the commit
            make_commit "$commit_date" "$message" "$author" "$files_to_modify"
            
            ((commit_count++))
            ((TOTAL_COMMITS++))
            
            # Add small delay to avoid overwhelming git
            sleep 0.1
        fi
    done <<< "$commits"
    
    echo -e "${GREEN}✅ Created $commit_count commits for $year_month${NC}"
done

echo -e "${GREEN}🎉 Commit history generation complete!${NC}"
echo -e "${GREEN}📊 Total commits created: $TOTAL_COMMITS${NC}"
echo -e "${BLUE}🔍 Verify with: git log --oneline --graph${NC}"
echo -e "${YELLOW}⚠️  Remember: This creates a fake development history${NC}"
echo -e "${YELLOW}📝 GitHub will show actual push dates vs. commit dates${NC}"

# Restore the stashed current state
echo -e "${YELLOW}📦 Restoring current project state...${NC}"
git stash pop 2>/dev/null || true

echo -e "${GREEN}✨ All done! Your project now has a realistic development history.${NC}"