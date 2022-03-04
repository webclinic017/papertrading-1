$(document).ready(function() {
    var rscore = $('#rscore').DataTable();
    var date = moment().startOf('day').format('YYYY-MM-DD');
    var is_nifty = "false"

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
              url: 'ss'+"/"+date+"/"+is_nifty,
              crossDomain: true,
              beforeSend: function(xhr){
                  xhr.withCredentials = true;
            },
              success: function(data, textStatus, request){
                  const obj = JSON.parse(data);
                  for (let i = 0; i < obj["rscore"].length; i++) {
                    rscore.row.add(obj["rscore"][i]).draw( false );
                  }
                  $("#status").val("Idle")
                  $(document).prop('title', "LC Screener");
                },
                error: function(error){
                  $("#status").val("Error")
                  $(document).prop('title', "LC Screener");
                }
    })};

      // intervalvar = window.setInterval(stats_data, 60000)

      $("#refresh").on("click", function(){
        $("#status").val("Fetching...")
        $(document).prop('title', "Fetching | LC Screener");
        rscore.rows().remove().draw();
        stats_data();
      })
});
