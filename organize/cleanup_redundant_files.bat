@echo off
echo Cleaning up redundant files...

:: Create a backup folder just in case
mkdir "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts"

:: Move redundant scripts to backup folder
echo Moving redundant scripts to backup folder...
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\distribution_fitting_backup.py" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\distribution_fitting_new.py" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\debug_distribution_fitting.py" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\test_delaynet.py" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\test_imports.py" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\test_network_analysis.py" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\analyze_missing_and_distribution.py" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\analyze_timeseries.py" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\run_nct_fitting.sh" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"

:: Move log files to backup folder
echo Moving log files to backup folder...
move "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\*.log" "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\backup_scripts\"

:: Create a list of essential files for GitHub
echo Creating list of essential files for GitHub...
echo # Essential Scripts and Results for GitHub Repository > "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo. >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo ## Primary Analysis Scripts: >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo validate_data.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo analyze_airports.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo network_analysis.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo distribution_fitting.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo. >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo ## Supporting Scripts: >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo load_data.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo load_dataframe.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo delaynet.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo network_metrics.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo eda_analysis.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo build_timeseries.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo detrend_timeseries.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo fast_distribution_fitting.py >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo. >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo ## Documentation: >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo README.md >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo Meeting_Report_Oct2025.md >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo. >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo ## Results folder with all visualizations and data files >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo. >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"
echo ## Raw data folder (only if size permits) >> "C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\files_for_github.txt"

echo Cleanup complete! Redundant files have been moved to the backup_scripts folder.
echo Essential files for GitHub are listed in files_for_github.txt
