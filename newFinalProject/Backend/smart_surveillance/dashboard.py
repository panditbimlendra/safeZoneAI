"""
Web Dashboard for Smart Surveillance System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path

def start_dashboard(system):
    """Start Streamlit dashboard"""
    st.set_page_config(
        page_title="Smart Surveillance Dashboard",
        page_icon="🎥",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-header">🎥 Smart Surveillance Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        if st.button("🔄 Refresh"):
            st.rerun()
        
        st.subheader("Camera Controls")
        camera_ids = system.camera_manager.get_camera_ids()
        selected_camera = st.selectbox("Select Camera", camera_ids)
        
        if st.button(f"📷 View Camera {selected_camera}"):
            # Show camera feed
            pass
        
        st.subheader("Alert Settings")
        alert_types = ["All", "Intrusion", "Loitering", "Abandoned Object", "Crowd", "Vehicle"]
        selected_alert_type = st.selectbox("Filter Alerts", alert_types)
        
        st.subheader("Export Data")
        if st.button("📊 Export Analytics"):
            # Export functionality
            pass
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Cameras", len(system.camera_manager.get_active_cameras()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Persons Detected", system.analytics.get('person_count', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Today's Alerts", system.analytics.get('alerts_today', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("System Uptime", "24h")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Feeds", "🚨 Alerts", "📈 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.header("Live Camera Feeds")
        
        cols = st.columns(2)
        for idx, camera_id in enumerate(camera_ids[:4]):
            with cols[idx % 2]:
                st.subheader(f"Camera: {camera_id}")
                # Placeholder for camera feed
                st.image("https://via.placeholder.com/640x360.png?text=Camera+Feed", 
                        use_column_width=True)
                
                # Camera stats
                stats = system.camera_manager.get_camera_stats(camera_id)
                st.write(f"FPS: {stats.get('fps', 'N/A')}")
                st.write(f"Status: {'✅ Active' if stats.get('active') else '❌ Inactive'}")
    
    with tab2:
        st.header("Recent Alerts")
        
        if system.alerts:
            alerts_df = pd.DataFrame(system.alerts[-10:])  # Last 10 alerts
            
            for _, alert in alerts_df.iterrows():
                with st.container():
                    alert_color = {
                        'high': '🔴',
                        'medium': '🟡', 
                        'low': '🟢'
                    }.get(alert.get('severity', 'low'), '⚪')
                    
                    st.markdown(f"""
                    <div class="alert-card">
                        <h4>{alert_color} {alert.get('type', 'Unknown').title()}</h4>
                        <p><strong>Camera:</strong> {alert.get('camera', 'N/A')}</p>
                        <p><strong>Time:</strong> {alert.get('timestamp', 'N/A')}</p>
                        <p><strong>Severity:</strong> {alert.get('severity', 'low')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No alerts in the last 24 hours")
        
        # Alert statistics
        st.subheader("Alert Statistics")
        if system.alerts:
            alert_types = pd.DataFrame(system.alerts)['type'].value_counts()
            fig = px.pie(values=alert_types.values, names=alert_types.index, 
                        title="Alert Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Person count over time
            st.subheader("Person Count Trend")
            # Generate sample data
            times = pd.date_range(start="today", periods=24, freq="H")
            counts = [system.analytics.get('person_count', 0)] * 24
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=counts, mode='lines', name='Person Count'))
            fig.update_layout(title="Hourly Person Count", xaxis_title="Time", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Zone activity
            st.subheader("Zone Activity")
            zone_data = {
                'Zone': ['Entrance', 'Parking', 'Lobby', 'Restricted'],
                'Alerts': [5, 3, 2, 8],
                'Activity': ['High', 'Medium', 'Low', 'Critical']
            }
            df_zones = pd.DataFrame(zone_data)
            fig = px.bar(df_zones, x='Zone', y='Alerts', color='Activity',
                        title="Alerts by Zone")
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.subheader("Activity Heatmap")
        # Generate sample heatmap data
        import numpy as np
        heatmap_data = np.random.rand(10, 10)
        fig = px.imshow(heatmap_data, title="Activity Heatmap", 
                       labels=dict(x="X Position", y="Y Position", color="Activity Level"))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("System Settings")
        
        with st.form("system_settings"):
            st.subheader("Detection Settings")
            confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
            iou = st.slider("IOU Threshold", 0.0, 1.0, 0.45)
            
            st.subheader("Alert Settings")
            email_alerts = st.checkbox("Enable Email Alerts", value=False)
            telegram_alerts = st.checkbox("Enable Telegram Alerts", value=True)
            
            st.subheader("Storage Settings")
            retention_days = st.number_input("Video Retention (days)", 1, 365, 7)
            max_storage = st.number_input("Max Storage (GB)", 10, 1000, 100)
            
            if st.form_submit_button("💾 Save Settings"):
                st.success("Settings saved successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown("### System Status: ✅ **Running**")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # For standalone dashboard testing
    st.title("Smart Surveillance Dashboard")
    st.info("Run main.py to start the surveillance system")