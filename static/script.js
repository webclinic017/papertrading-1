$(function(){
  var img = new Image();

  var end = moment().startOf('day').add(15, 'hour').add(30, 'minute');
  var start = end.clone().subtract(5, 'day').subtract(6, 'hour').subtract(16, 'minute');
  var current = start.clone();
  var tradingsymbol = 'NIFTY 50,RELIANCE,HDFCBANK'
  var refresh = false
  var game_mode = false
  var buy = true
  var entry = 0
  var stoploss = 0
  var target = 0
  var lookback = 7
  var leadhr = 0
  var gameid = "unclassified"
  var intervalvar = null
  var tick_mode = false

  $('#current-date').val(current.toString());
  $('#lookback').val(lookback);
  $('#tradingsymbol').val(tradingsymbol);
  $('#sl').val(stoploss);
  $('#Entry').val(entry);
  $('#Target').val(target);
  $('#leadhr').val(leadhr);

  // clear ui
  $('#game-control').hide();
  $('#statdiv').hide();
  $('#imgdiv').attr('class', 'col-sm-12 my-custom-scrollbar');
  $('.my-custom-scrollbar').css('min-height', '700px');
  // wheelzoom(document.querySelector('img.zoom'));


  $("#game-mode").change(function(){
      if($(this).is(":checked")) {
        $('#game-control').show()
        $('#statdiv').show()
        $('#imgdiv').attr('class', 'col-sm-10 my-custom-scrollbar')
        $('.my-custom-scrollbar').css('min-height', '580px')
        game_mode = true
      } else {
        $('#game-control').hide()
        game_mode = false
        $('#statdiv').hide()
        $('#imgdiv').attr('class', 'col-sm-12 my-custom-scrollbar')
        $('.my-custom-scrollbar').css('min-height', '700px')

      }
  })

  $("#game-stats").change(function(){
      if($(this).is(":checked")) {
        $('#statdiv').show()
        $('#imgdiv').attr('class', 'col-sm-10 my-custom-scrollbar')
      } else {
        $('#statdiv').hide()
        $('#imgdiv').attr('class', 'col-sm-12 my-custom-scrollbar')
      }
  })

  $("#tick-mode").change(function(){
      if($(this).is(":checked")) {
        tick_mode = true
        $("#tradingsymbol").val("RELIANCE");
        tradingsymbol = "RELIANCE"
      } else {
        tick_mode = false
        $("#tradingsymbol").val("NIFTY 50,RELIANCE,HDFCBANK");
        tradingsymbol = "NIFTY 50,RELIANCE,HDFCBANK"
      }
  })

  img.onload = function() {
    console.log(this.width + 'x' + this.height);
    height = parseInt(this.height);
    width = parseInt(this.width);
    var maxheight = parseInt($(".img-style").css('max-height'));
    var newwidth = maxheight > height ? width : width * (maxheight/height);
    $('div.h-cross').css('width', newwidth + "px");
  }


  $("#buy").change(function(){
      if($(this).is(":checked")) {
        buy = true
      } else {
        buy = false
      }
  })

  function rrr_refresh(){
    var rrr = (parseFloat(target) - parseFloat(entry))/ (parseFloat(stoploss) - parseFloat(entry));
    rrr = Math.abs(rrr)
    rrr = rrr.toFixed(2)
    $('#rrr').val(rrr);
  }

  $("#tradingsymbol").change(function(){
    tradingsymbol = $(this).val()
  })

  $("#Entry").change(function(){
    entry = $(this).val();
    rrr_refresh();
  })

  $("#Target").change(function(){
    target = $(this).val()
    rrr_refresh();
  })

  $("#sl").change(function(){
    stoploss = $(this).val()
    rrr_refresh();
  })

  $("#lookback").change(function(){
    lookback = $(this).val()
  })

  $("#gameid").change(function(){
    gameid = $(this).val()
  })

  $('#imgdiv').on('mousemove', function(e){
    $('div.h-cross').css('top', e.clientY);
  })

  $('#imgdiv').on('mouseenter', function(e){
    $('div.h-cross').css('visibility', 'visible');
  })

  $('#imgdiv').on('mouseleave', function(e){
    $('div.h-cross').css('visibility', 'hidden');
  })

  $('#clear-notes').on('click', function(){
    $('#notes').val('')
  })

  function stats_refresh(){
          $.ajax({
            headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
            type: 'GET',
            url: 'stats/'+gameid,
            crossDomain: true,
            beforeSend: function(xhr){
                xhr.withCredentials = true;
          },
            success: function(data, textStatus, request){
                const obj = JSON.parse(data);
                $('#gameidview').val(obj["gameid"]);
                $('#accuracy').val(obj["accuracy"]);
                $('#change').val(obj["change"]);
                $('#count').val(obj["count"]);
                $('#maxdrawdown').val(obj["maxdrawdown"]);
                $('#partial').val(obj["partial"]);
                $('#peak').val(obj["peak"]);
            }
          });
        }

  function day_change(){
    $.ajax({
      headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
      type: 'GET',
      url: 'mis',
      crossDomain: true,
      beforeSend: function(xhr){
          xhr.withCredentials = true;
    },
      success: function(data, textStatus, request){
            data = JSON.parse(data);
            $("#dchange").val(data["dchange"]);
          },
      error: function(error){
        alert("Error, Failed to load graph")
      }
    });
  }

  function next_step(t1, t2, update){
    update = update ? 1: 0
    tm = tick_mode ? 1: 0
    lead = $("#lead").val()

    $.ajax({
      headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
      type: 'GET',
      url: 'hgraph/'+tradingsymbol+'/'+t1+'/'+t2+"/"+update+"/"+tm+"/"+lead,
      crossDomain: true,
      beforeSend: function(xhr){
          xhr.withCredentials = true;
    },
      success: function(data, textStatus, request){
          data = JSON.parse(data);
          $("#graph-image").attr('src', data["imgurl"]);
          img.src = data["imgurl"];
          if(data["update"] == "1"){
            $('#Entry').val(data["close"]);
            entry = data["close"];
            $('#Target').val(data["close"]);
            target = data["close"];
            $('#sl').val(data["close"]);
            stoploss = data["close"];
            rrr_refresh();
          }

      },
      error: function(error){
        alert("Error, Failed to load graph")
      }
    });
  }

  function schedule_refresh(){
    // console.log('refreshing');
    next_step(start.toString(), end.toString(), false);
    $('#last-updated').val(moment().format('MMMM Do YYYY, h:mm:ss a'));
    day_change();
  }

  $("#repeat").change(function(){
      if($(this).is(":checked")) {
        intervalvar = window.setInterval(schedule_refresh, 60000)
        console.log("starting refresh");
      } else {
        window.clearInterval(intervalvar)
        console.log('cleared refresh');
      }
  })


  $("#graph-history").click(function(){
    next_step(start.toString(), end.toString(), false);
    $(document).prop('title', tradingsymbol + " | Lumbinicapital");
  });


  $('input[name="date-range"]').daterangepicker({
      timePicker: true,
      startDate: start,
      endDate: end,
      locale: {
        format: 'DD/MM/YYYY hh:mm A'
      }
    }, function(start_date, end_date){
      start = start_date
      end = end_date
      current = start_date.clone()
      $('#current-date').val(current.toString())
    });


  $('#Next-Hour').on('click', function(){
    current = current.add(30, 'minute');
    $('#current-date').val(current.toString());

    var t1 = current.clone().subtract(lookback, "day").startOf('day').toString();
    var t2 = current.toString();

    next_step(t1, t2, true)

    leadhr = leadhr + 1;
    $(document).prop('title', tradingsymbol + " | Lumbinicapital");
    $('#leadhr').val(leadhr);
  });


  $('#Review-Bet').on('click', function(){
    var fwd = 1
    var intraday = 1
    var bint = buy ? 1: 0

    var keyvar = {
      "name": tradingsymbol,
      "start": current.startOf('day').toString(),
      "fwd": fwd,
      "intraday": intraday,
      "leadhr": leadhr,
      "entry": entry,
      "stoploss": stoploss,
      "target": target,
      "buy": bint,
      "gameid": gameid
    }

    $.ajax({
      headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
      type: 'POST',
      data: keyvar,
      url: 'outcome',
      crossDomain: true,
      beforeSend: function(xhr){
          xhr.withCredentials = true;
    },
      success: function(data, textStatus, request){
          $("#Change").val(data.change)
      }
    });
  });


  $('#reveal').on('click', function(){
    var t1 = current.clone().subtract(lookback, "day").startOf('day').toString();
    var t2 = current.clone().add(17, "hour").toString();
    next_step(t1, t2, false);
    stats_refresh();
  });


  $('#Next-Day').click(function(){
    var increment = 1
    if (current.format('E') == 5){
      increment = 3
    } else if (current.format('E') == 6) {
      increment = 2
    }
    current = current.add(increment, 'day').startOf('day').add(9, 'hour').add(14, 'minute');
    leadhr = 0;
    $('#current-date').val(current.toString());
    $('#leadhr').val(leadhr);
    $("#Change").val('undefined');
    $('#Entry').val(0);
    $('#sl').val(0);
    $('#Target').val(0);
    $(document).prop('title', tradingsymbol + " | Lumbinicapital");
    stats_refresh();
  });

  $('#Prev-Day').click(function(){
    var decrement = 1
    if (current.format('E') == 1){
      decrement = 3
    } else if (current.format('E') == 6) {
      decrement = 2
    }
    current = current.subtract(decrement, 'day').startOf('day').add(9, 'hour').add(14, 'minute');
    leadhr = 0;
    $('#current-date').val(current.toString());
    $('#leadhr').val(leadhr);
    $("#Change").val('undefined');
    $('#Entry').val(0);
    $('#sl').val(0);
    $('#Target').val(0);
    $(document).prop('title', tradingsymbol + " | Lumbinicapital");
    stats_refresh();
  });

  stats_refresh();


});
