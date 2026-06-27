from datetime import datetime
from typing import Any, Dict, List

import altair as alt
import streamlit as st

try:
    from dashboard.mock_data import build_alerts, build_people, build_sensor_window
except ModuleNotFoundError:
    from mock_data import build_alerts, build_people, build_sensor_window


PENDING_STATUS = "Awaiting response"


def configure_page() -> None:
    st.set_page_config(
        page_title="Fall alerts",
        page_icon="⚠️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1100px; padding-top: 1.8rem;}
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialise_state() -> None:
    if "mock_seed" not in st.session_state:
        st.session_state.mock_seed = 7
    if "people" not in st.session_state:
        st.session_state.people = build_people()
    if "alerts" not in st.session_state:
        refresh_mock_data()


def refresh_mock_data() -> None:
    seed = st.session_state.get("mock_seed", 7)
    st.session_state.alerts = build_alerts(seed)
    st.session_state.mock_seed = seed + 1
    st.session_state.last_refresh = datetime.now()


def format_datetime(value: datetime) -> str:
    return value.strftime("%d/%m/%Y %H:%M")


def person_by_id(client_id: str) -> Dict[str, Any]:
    return next(person for person in st.session_state.people if person["client_id"] == client_id)


def update_alert(alert_id: str, status: str) -> None:
    for alert in st.session_state.alerts:
        if alert["id"] == alert_id:
            alert["status"] = status
            return


def create_sos_alert(person: Dict[str, Any]) -> None:
    st.session_state.alerts.insert(
        0,
        {
            "id": "SOS-{0}".format(datetime.now().strftime("%H%M%S")),
            "detected_at": datetime.now().replace(microsecond=0),
            "client_id": person["client_id"],
            "dataset": person["dataset"],
            "person": person["person"],
            "caregiver": person["caregiver"],
            "caregiver_phone": person["caregiver_phone"],
            "confidence": None,
            "status": "Response activated",
            "source": "SOS button",
        },
    )


def set_consent(client_id: str, consent: bool) -> None:
    person = person_by_id(client_id)
    person["consent"] = consent
    person["device_status"] = "Protection active" if consent else "Consent missing"


def alerts_for_person(client_id: str) -> List[Dict[str, Any]]:
    return [alert for alert in st.session_state.alerts if alert["client_id"] == client_id]


def render_sidebar() -> str:
    st.sidebar.title("Fall alerts")
    st.sidebar.caption("Prototype · demo mode")
    st.sidebar.divider()
    query_view = st.query_params.get("view", "User")
    default_index = 1 if query_view == "Caregiver" else 0
    view = st.sidebar.radio("View", ("User", "Caregiver"), index=default_index)
    st.sidebar.divider()
    if st.sidebar.button("Reset mock data", width="stretch"):
        refresh_mock_data()
        st.rerun()
    st.sidebar.caption(
        "Last update: {0}".format(
            st.session_state.last_refresh.strftime("%H:%M:%S")
        )
    )
    return view


def render_user_fall_prompt(alert: Dict[str, Any]) -> None:
    st.error("We detected a possible fall. Are you okay?")
    st.caption("Your response helps avoid unnecessary alerts to your caregiver.")
    left, right = st.columns(2)
    if left.button("I'm okay", type="primary", width="stretch"):
        update_alert(alert["id"], "False positive")
        st.rerun()
    if right.button("I need help", width="stretch"):
        update_alert(alert["id"], "Response activated")
        st.rerun()


def render_user_history(alerts: List[Dict[str, Any]]) -> None:
    st.subheader("Recent history")
    rows = [
        {
            "Date": format_datetime(alert["detected_at"]),
            "Source": alert["source"],
            "Status": alert["status"],
        }
        for alert in alerts[:6]
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def sensor_chart(
    samples: List[Dict[str, Any]], field: str, title: str, unit: str
) -> alt.LayerChart:
    base = alt.Chart(alt.Data(values=samples)).encode(
        x=alt.X("second:Q", title="Seconds"),
    )
    fall_area = base.mark_rect(opacity=0.12, color="#dc2626").encode(
        x="second:Q",
        x2=alt.X2("second_end:Q"),
    ).transform_filter(alt.datum.zone == "Possible fall")
    signal = base.mark_line(strokeWidth=3).encode(
        y=alt.Y("{0}:Q".format(field), title=unit),
        color=alt.Color(
            "zone:N",
            scale=alt.Scale(
                domain=["Normal", "Possible fall"],
                range=["#16a34a", "#dc2626"],
            ),
            legend=None,
        ),
    )
    return (fall_area + signal).properties(width="container", height=190, title=title)


def render_sensor_signals(person: Dict[str, Any], possible_fall: bool) -> None:
    st.caption("Local window for the last 60 seconds · simulated data.")
    samples = build_sensor_window(person["client_id"], possible_fall)
    left, right = st.columns(2)
    with left:
        st.altair_chart(
            sensor_chart(samples, "acceleration", "Acceleration", "Magnitude (g)"),
        )
    with right:
        st.altair_chart(
            sensor_chart(samples, "rotation", "Rotation", "Magnitude (°/s)"),
        )


def render_user_view() -> None:
    st.title("My protection")
    person_names = {person["person"]: person["client_id"] for person in st.session_state.people}
    selected_name = st.selectbox("Demo user", person_names)
    person = person_by_id(person_names[selected_name])
    personal_alerts = alerts_for_person(person["client_id"])
    pending = next(
        (alert for alert in personal_alerts if alert["status"] == PENDING_STATUS), None
    )

    if pending:
        render_user_fall_prompt(pending)

    st.subheader("My status")
    render_sensor_signals(person, pending is not None)

    with st.container(border=True):
        st.subheader("Consent")
        st.caption("Allow local data to be used for fall detection.")
        consent = st.toggle("Protection and detection active", value=person["consent"])
        if consent != person["consent"]:
            set_consent(person["client_id"], consent)
            st.rerun()

    st.subheader("Do you need help?")
    st.caption("Use the SOS button to alert your caregiver immediately.")
    if st.button("SOS · Request help", type="primary", width="stretch"):
        create_sos_alert(person)
        st.success("Request sent to your caregiver.")

    st.write("")
    render_user_history(personal_alerts)


def render_live_alerts(alerts: List[Dict[str, Any]]) -> None:
    active = [alert for alert in alerts if alert["status"] == PENDING_STATUS]
    st.subheader("Real-time alerts")
    if not active:
        st.info("There are no alerts awaiting a response.")
        return

    for alert in active:
        with st.container(border=True):
            details, actions = st.columns([3, 2])
            with details:
                st.markdown("#### Possible fall · {0}".format(alert["person"]))
                st.write(
                    "**{0}**".format(format_datetime(alert["detected_at"]))
                )
                if alert["confidence"] is not None:
                    st.caption("Model confidence: {0:.0f}%".format(alert["confidence"] * 100))
            with actions:
                st.write("")
                if st.button("Confirm contact", key="confirm-{0}".format(alert["id"]), width="stretch"):
                    update_alert(alert["id"], "Confirmed")
                    st.rerun()
                if st.button("Activate response", key="respond-{0}".format(alert["id"]), type="primary", width="stretch"):
                    update_alert(alert["id"], "Response activated")
                    st.rerun()


def render_people(people: List[Dict[str, Any]]) -> None:
    st.subheader("People under your care")
    rows = [
        {
            "Person": person["person"],
            "Relationship": person["relationship"],
            "Status": person["device_status"],
            "Consent": "Active" if person["consent"] else "Missing",
        }
        for person in people
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def render_caregiver_history(alerts: List[Dict[str, Any]]) -> None:
    st.subheader("Recent history")
    rows = [
        {
            "Date": format_datetime(alert["detected_at"]),
            "Person": alert["person"],
            "Source": alert["source"],
            "Status": alert["status"],
        }
        for alert in alerts
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def render_caregiver_view() -> None:
    st.title("Caregiver area")
    st.caption("Monitor associated people and respond to fall alerts.")
    caregivers = sorted({person["caregiver"] for person in st.session_state.people})
    caregiver = st.selectbox("Demo caregiver", caregivers)
    people = [
        person for person in st.session_state.people if person["caregiver"] == caregiver
    ]
    client_ids = {person["client_id"] for person in people}
    alerts = [
        alert for alert in st.session_state.alerts if alert["client_id"] in client_ids
    ]
    render_live_alerts(alerts)
    st.write("")
    render_people(people)
    st.write("")
    render_caregiver_history(alerts)


def main() -> None:
    configure_page()
    initialise_state()
    view = render_sidebar()
    if view == "User":
        render_user_view()
    else:
        render_caregiver_view()


if __name__ == "__main__":
    main()
