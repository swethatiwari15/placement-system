"""
Reusable Streamlit Components
Modern card-based UI components
"""

import streamlit as st
from typing import Any, Callable, Optional, Dict, List
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def render_metric_card(title: str, value: str, description: str = "", 
                      color: str = "#1f77b4", icon: str = "📊"):
    """Render a metric card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
        border-left: 4px solid {color};
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    ">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">
            {icon} {title}
        </div>
        <div style="font-size: 32px; font-weight: bold; color: {color};">
            {value}
        </div>
        {f'<div style="font-size: 12px; color: #999; margin-top: 5px;">{description}</div>' if description else ''}
    </div>
    """, unsafe_allow_html=True)


def render_card(title: str, content: Any, color: str = "#1f77b4"):
    """Render a card container."""
    st.markdown(f"""
    <div style="
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    ">
        <div style="
            border-bottom: 2px solid {color};
            padding-bottom: 10px;
            margin-bottom: 15px;
        ">
            <h3 style="margin: 0; color: {color};">{title}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if callable(content):
        content()
    else:
        st.write(content)
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_result(placed: bool, probability: float, 
                            confidence_level: str, features: Dict[str, Any]):
    """Render prediction result with visual indicators."""
    
    # Main result card
    if placed:
        bg_color = "#d4edda"
        border_color = "#2ecc71"
        text_color = "#155724"
        status = "✓ PLACED"
        status_emoji = "🎉"
    else:
        bg_color = "#f8d7da"
        border_color = "#e74c3c"
        text_color = "#721c24"
        status = "✗ NOT PLACED"
        status_emoji = "💡"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="
            background: {bg_color};
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 25px;
            text-align: center;
        ">
            <div style="font-size: 36px; margin-bottom: 10px;">{status_emoji}</div>
            <div style="font-size: 28px; font-weight: bold; color: {text_color};">
                {status}
            </div>
            <div style="font-size: 16px; color: {text_color}; margin-top: 10px;">
                Probability: <strong>{probability*100:.1f}%</strong>
            </div>
            <div style="font-size: 14px; color: {text_color}; margin-top: 5px;">
                Confidence: <strong>{confidence_level}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 33], 'color': "#f8f9f9"},
                    {'range': [33, 66], 'color': "#f0f3f4"},
                    {'range': [66, 100], 'color': "#ebf5fb"}
                ],
                'threshold': {
                    'line': {'color': "#e74c3c", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_feature_comparison(features: Dict[str, float], 
                             feature_ranges: Dict[str, tuple],
                             feature_labels: Dict[str, str]):
    """Render feature values compared to ranges."""
    
    data = []
    for feature, value in sorted(features.items()):
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            percentile = (value - min_val) / (max_val - min_val) * 100
            data.append({
                'Feature': feature_labels.get(feature, feature),
                'Value': value,
                'Percentile': percentile
            })
    
    df = pd.DataFrame(data).sort_values('Percentile', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['Feature'],
        x=df['Percentile'],
        orientation='h',
        marker=dict(
            color=df['Percentile'],
            colorscale='RdYlGn',
            showscale=False,
            line=dict(width=0)
        ),
        text=df['Value'].apply(lambda x: f'{x:.1f}'),
        textposition='outside',
        hovertemplate='%{y}<br>Value: %{text}<br>Percentile: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Performance Across Features",
        xaxis_title="Percentile Score (%)",
        yaxis_title="",
        height=400,
        showlegend=False,
        margin=dict(l=150)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_insights(insights: Dict[str, Any]):
    """Render prediction insights."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_metric_card(
            "Prediction",
            insights['prediction'],
            "Overall placement likelihood",
            "#1f77b4",
            "📌"
        )
    
    with col2:
        render_metric_card(
            "Confidence",
            insights['confidence'],
            f"{insights['probability']} probability",
            "#f39c12",
            "📊"
        )
    
    with col3:
        render_metric_card(
            "Status",
            "Positive" if "Placed" in insights['prediction'] else "Needs Work",
            "Overall assessment",
            "#2ecc71" if "Placed" in insights['prediction'] else "#e74c3c",
            "✓" if "Placed" in insights['prediction'] else "⚠"
        )
    
    # Strong and weak factors
    if insights['strong_factors'] or insights['weak_factors']:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if insights['strong_factors']:
                st.markdown("### 💪 Strong Factors")
                for factor in insights['strong_factors']:
                    st.markdown(f"""
                    <div style="
                        background: #d4edda;
                        padding: 10px;
                        border-radius: 6px;
                        margin-bottom: 8px;
                        border-left: 4px solid #2ecc71;
                    ">
                        <strong>{factor['factor']}</strong>: {factor['value']} ({factor['rating']})
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            if insights['weak_factors']:
                st.markdown("### ⚠️ Areas for Improvement")
                for factor in insights['weak_factors']:
                    st.markdown(f"""
                    <div style="
                        background: #f8d7da;
                        padding: 10px;
                        border-radius: 6px;
                        margin-bottom: 8px;
                        border-left: 4px solid #e74c3c;
                    ">
                        <strong>{factor['factor']}</strong>: {factor['value']} ({factor['rating']})
                    </div>
                    """, unsafe_allow_html=True)
    
    # Recommendations
    if insights['recommendations']:
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        for i, rec in enumerate(insights['recommendations'], 1):
            st.markdown(f"**{i}.** {rec}")


def render_confusion_matrix(confusion_matrix: List[List[int]]):
    """Render confusion matrix visualization."""
    import pandas as pd
    
    df = pd.DataFrame(
        confusion_matrix,
        index=['Not Placed', 'Placed'],
        columns=['Predicted Not Placed', 'Predicted Placed']
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='Blues',
        text=df.values,
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='%{y} / %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        height=350,
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_metrics_table(metrics: Dict[str, Any]):
    """Render model metrics in table format."""
    import pandas as pd
    
    data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Score': [
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('auc_roc', 0):.4f}"
        ]
    }
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_feature_importance_chart(importance: Dict[str, float]):
    """Render feature importance bar chart."""
    import pandas as pd
    
    df = pd.DataFrame(
        list(importance.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        title="Feature Importance Ranking"
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=150)
    )
    
    st.plotly_chart(fig, use_container_width=True)

