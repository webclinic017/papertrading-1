$(document).ready(function() {

    var options = {
        "searching": false,
        "orderMulti": true,
        "pageLength": 10,
        "fixedHeader": false
      };

    var base_options = Object.assign({},
        options,
        {"createdRow": function( row, data, dataIndex ) {
                        ts = encodeURIComponent(data[0].trim());
                        $('td:eq(0)', row).html( `<a target="_blank" href=/?tradingsymbol=${ts}&autoload=1>${data[0]}</a>` );

                        if ( data[3] > 0 && data[4] > 0 ){
                          $(row).addClass( 'yellow-row' );
                        } else if ( data[3] > 0 ) {
                          $(row).addClass( 'green-row' );
                        } else if (data[4] > 0) {
                          $(row).addClass( 'red-row' );
                        }
                      }})
  var ms_options = Object.assign({},
      options,
      {"createdRow": function( row, data, dataIndex ) {
                      ts = encodeURIComponent(data[0].trim());
                      $('td:eq(0)', row).html( `<a target="_blank" href=/?tradingsymbol=${ts}>${data[0]}</a>` );

                      if ( data[8] > 0.5 &&  data[8] < 2){
                        $(row).addClass( 'yellow-row' );
                      } else if ( data[8] >= 2 ) {
                        $(row).addClass( 'green-row' );
                      } else if (data[8] <= 0.5) {
                        $(row).addClass( 'red-row' );
                      }
                    } });

    var extended_sector = $('#extended-sector').DataTable(Object.assign({}, base_options, {"order": [[ 6, "desc" ]]} ));
    var extended_stock = $('#extended-stock').DataTable(Object.assign({}, base_options, {"order": [[ 7, "desc" ]]} ));
    var extended_stock_down = $('#extended-stock-down').DataTable(Object.assign({}, base_options, {"order": [[ 7, "desc" ]]} ));
    var extended_stock_up = $('#extended-stock-up').DataTable(Object.assign({}, base_options, {"order": [[ 7, "desc" ]]} ));
    var market_screeners = $('#market-screeners').DataTable(Object.assign({}, ms_options, {"order": [[ 14, "desc" ]]} ));
    var no_extended_stock = $('#no-extended-stock').DataTable(Object.assign({}, base_options, {"order": [[ 9, "desc" ]]} ));

    var tables = [{"name": "extended_sector", "var": extended_sector},
                  {"name": "extended_stock", "var": extended_stock},
                  {"name": "extended_stock_down", "var": extended_stock_down},
                  {"name": "extended_stock_up", "var": extended_stock_up},
                  {"name": "market_screeners", "var": market_screeners},
                  {"name": "no_extended_stock", "var": no_extended_stock}]

    var date = moment().startOf('day').format('YYYY-MM-DD');
    var is_nifty = "true"

    $('#date').val(date.toString());
    $("#date").change(function(){
      date = $(this).val()
      console.log(date);
    })
    $("#is_nifty").change(function(){
        if($(this).is(":checked")) {
          is_nifty = "true"
        } else {
          is_nifty = "false"
        }
    })

    function stats_data(){
            $.ajax({
              headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
              type: 'GET',
              url: "events_data"+"/"+date+"/"+is_nifty,
              crossDomain: true,
              beforeSend: function(xhr){
                  xhr.withCredentials = true;
            },
              success: function(data, textStatus, request){
                  const obj = JSON.parse(data);
                  for (let j = 0; j < tables.length; j++){

                    var key = tables[j]["name"];
                    tables[j]["var"].rows().remove().draw();

                    for (let i = 0; i < obj[key].length; i++) {
                      tables[j]["var"].row.add(obj[key][i]).draw( false );
                    }

                  }

                  $("#status").val("Idle")
                  $(document).prop('title', "LC Events");
                },
                error: function(error){
                  $("#status").val("Error")
                  $(document).prop('title', "LC Events");
                }
    })};

      // intervalvar = window.setInterval(stats_data, 60000)

      function schedule_refresh(){
        $("#status").val("Fetching...")
        $(document).prop('title', "Fetching | LC Events");
        stats_data();
      }

      $("#refresh").on("click", function(){
        schedule_refresh();
      });

      $("#repeat").change(function(){
          if($(this).val() > 20) {
            intervalvar = window.setInterval(schedule_refresh, $(this).val() * 1000)
            console.log("starting refresh");
            schedule_refresh();
          } else {
            window.clearInterval(intervalvar)
            console.log('cleared refresh');
          }
      });
});
