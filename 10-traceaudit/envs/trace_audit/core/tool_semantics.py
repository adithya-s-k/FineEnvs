"""Explicit tau2 DB mutators; unknown tools must not be guessed from their names."""

TAU2_WRITE_TOOLS = {
    "airline": frozenset(
        {
            "book_reservation",
            "cancel_reservation",
            "update_reservation_flights",
            "update_reservation_baggages",
            "update_reservation_passengers",
        }
    ),
    "retail": frozenset(
        {
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "return_delivered_order_items",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
        }
    ),
}
TAU2_READ_TOOLS = {
    "airline": frozenset(
        {
            "get_user_details",
            "get_reservation_details",
            "search_direct_flight",
            "search_onestop_flight",
            "list_all_airports",
            "calculate",
            "transfer_to_human_agents",
        }
    ),
    "retail": frozenset(
        {
            "get_user_details",
            "get_order_details",
            "get_product_details",
            "get_item_details",
            "find_user_id_by_name_zip",
            "find_user_id_by_email",
            "calculate",
            "transfer_to_human_agents",
        }
    ),
}
